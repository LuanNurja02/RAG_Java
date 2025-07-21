import os
import time
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


# Carica documenti PDF
parser = PyMuPDFReader()
documents = SimpleDirectoryReader("DATA/book", file_extractor={".pdf": parser}).load_data()
print(f"Caricati {len(documents)} documenti")

# Imposta il template dei metadata
for doc in documents:
    doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"
    if "page_label" not in doc.excluded_embed_metadata_keys:
        doc.excluded_embed_metadata_keys.append("page_label")

# LLM e embedding model
llm = Ollama(
    model="llama3.1:8b",
    temperature=0.1,
    max_tokens=14000,
    request_timeout=600,
    context_window=14000,
    streaming=False,
    min_length=100,
    top_p=0.9,
    repeat_penalty=1.2
)

embed_model = HuggingFaceEmbedding(
    model_name="intfloat/e5-small-v2"
)

# Pipeline: split, estrai titoli e domande
text_splitter = SentenceSplitter(
    
    chunk_size=1000, 
    chunk_overlap=200
    
    )
title_extractor = TitleExtractor(
    
    llm=llm,
    nodes=3
    
    )
qa_extractor = QuestionsAnsweredExtractor(
    
    llm=llm,
    questions=2, 

    
    )

pipeline = IngestionPipeline(
    transformations=[
        
        text_splitter, 
        title_extractor, 
        qa_extractor
        
        ]
)

nodes = pipeline.run(
    
    documents=documents, 
    show_progress=True, 
    in_place=True
    
    )
print("✅ Pipeline completata completamente in locale.")

# Pinecone setup
os.environ["PINECONE_API_KEY"] = "pcsk_6VBs3G_8DQhyP34krGmTda5APDdDBnA849MLsswfmSukUhB4Ct1CaUsPvPWHGCVSuXdy5T"
api_key = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)
index_name = "meta-lib"

# Crea indice se non esiste
if index_name not in list(pc.list_indexes()):
    print(f"Indice '{index_name}' non trovato, lo creo...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Indice creato")
    time.sleep(10)
else:
    print(f"Indice '{index_name}' trovato")


pinecone_index = pc.Index(index_name)

# Controlla contenuto
index_stats = pinecone_index.describe_index_stats()
print(f"Statistiche indice PRIMA: {index_stats}")

# Costruisci Vector Store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Crea o carica l’indice
if index_stats['total_vector_count'] == 0:
    print("Indice vuoto, creo index dai nodes...")
    index = VectorStoreIndex(
        
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
        
    )
    print(f"Index creato con {len(nodes)} nodes")
else:
    print("Indice esistente, lo carico...")
    index = VectorStoreIndex.from_vector_store(
        
        vector_store=vector_store,
        embed_model=embed_model
        
    )
    print("Index caricato")
