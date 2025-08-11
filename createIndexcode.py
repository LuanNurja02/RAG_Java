import os
import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import  SummaryExtractor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from tree_sitter import Parser
from dotenv import load_dotenv
load_dotenv(dotenv_path='pinecone_key.env')

JAVA_CODEBASE_PATH = "DATA/codebase"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_INDEX_NAME = "codebase-java" 


# Parametri del modello di embedding
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
EMBEDDING_DIMENSION = 768 

# Parametri di CodeSplitter
CODE_CHUNK_LINES = 500
CODE_CHUNK_OVERLAP = 100


print(f"Caricamento dei documenti Java da: {JAVA_CODEBASE_PATH}...")
documents = SimpleDirectoryReader(
    JAVA_CODEBASE_PATH,
    recursive=True, 
    required_exts=[".java"] #solo i file .java
).load_data()
print(f"Caricati {len(documents)} documenti Java.")




print("Inizializzazione del modello di embedding (HuggingFaceEmbedding)...")
embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME
)



print("Configurazione della pipeline di ingestion...")

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

# CodeSplitter per dividere il codice Java (basato su tree-sitter per java)
code_splitter = CodeSplitter(
    
    language="java",
    chunk_lines=CODE_CHUNK_LINES,
    chunk_lines_overlap=CODE_CHUNK_OVERLAP
    
)



code_explainer = SummaryExtractor(
    llm=llm,
    nodes=3,
)



# Definizione della pipeline 
pipeline = IngestionPipeline(
    transformations=
    [
        code_splitter,
        code_explainer
        
    ]
)



print("Esecuzione della pipeline di ingestion per generare i nodi...")
nodes = pipeline.run(
    
    documents=documents,
    show_progress=True,
    in_place=True 
    
)
print(f"Pipeline completata! Generati {len(nodes)} nodi di codice.")



print(f"Configurazione di Pinecone per l'indice '{PINECONE_INDEX_NAME}'...")
try:
    api_key = os.environ["PINECONE_API_KEY"]
except KeyError:
    print("Errore: La variabile d'ambiente 'PINECONE_API_KEY' non è impostata.")
    print("Assicurati di impostarla prima di eseguire lo script.")
    exit()

pc = Pinecone(api_key=api_key)

# Controlla se l'indice esiste, altrimenti crealo
if PINECONE_INDEX_NAME not in list(pc.list_indexes()):
    print(f"Indice '{PINECONE_INDEX_NAME}' non trovato, lo creo...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION, 
        metric="cosine", 
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )
    print("Indice creato. Attendendo che sia pronto...")
    time.sleep(10) 
else:
    print(f"Indice '{PINECONE_INDEX_NAME}' trovato.")

pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Controlla le statistiche dell'indice prima di aggiungere
index_stats_before = pinecone_index.describe_index_stats()
print(f"Statistiche indice Pinecone PRIMA dell'ingestion: {index_stats_before}")


vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Crea un nuovo indice se è vuoto, altrimenti carica quello esistente
if index_stats_before['total_vector_count'] == 0:
    print("L'indice è vuoto. Creazione di un nuovo indice vettoriale da zero...")
    index = VectorStoreIndex(
        
        nodes, # I nodi generati dalla pipeline
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
        
    )
    print(f"Indice della codebase Java creato con successo con {len(nodes)} nodi.")
else:
    print("L'indice esiste e contiene dati. Caricamento dell'indice esistente...")
    
    index = VectorStoreIndex.from_vector_store(
        
        vector_store=vector_store,
        embed_model=embed_model
        
    )
    print("Indice della codebase Java caricato.")

# Controlla le statistiche dell'indice dopo l'aggiunta
index_stats_after = pinecone_index.describe_index_stats()
print(f"Statistiche indice Pinecone DOPO l'ingestion: {index_stats_after}")
print("\nProcesso di indicizzazione completato!")

