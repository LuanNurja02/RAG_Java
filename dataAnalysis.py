
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import SimpleDirectoryReader
import re
from llama_index.readers.file import MarkdownReader

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

# Retrievers
from llama_index.core.retrievers import (
    VectorIndexRetriever,
)
from sentence_transformers import SentenceTransformer
from llama_index.core.chat_engine import ContextChatEngine 
from llama_index.core.memory import ChatMemoryBuffer
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
import time
import threading
import sys
import torch
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PDFReader,
    PyMuPDFReader
 
)
import os
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding

from llama_index.llms.ollama import Ollama
# PyMuPDF Reader example

md_parser = MarkdownReader()
parser = PyMuPDFReader()
file_extractor = {
    ".pdf": parser,
    ".md": md_parser,
}
documents = SimpleDirectoryReader(
    "DATA/java_api", file_extractor=file_extractor
).load_data()


# Inizializza il parser per dividere i documenti in chunk
text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100,include_metadata=False)

# Applica il chunking ai documenti
nodes = text_splitter.get_nodes_from_documents(documents) #questo node contiene il testo diviso in chunk


# Set up Pinecone API key
os.environ["PINECONE_API_KEY"] = "pcsk_6VBs3G_8DQhyP34krGmTda5APDdDBnA849MLsswfmSukUhB4Ct1CaUsPvPWHGCVSuXdy5T"
api_key = os.environ["PINECONE_API_KEY"]

# Create Pinecone Vector Store
pc = Pinecone(api_key=api_key)

index_name = "java-api"
# Controlla se l'indice esiste già
if index_name not in pc.list_indexes():
    print(f"Indice '{index_name}' non trovato, lo creo...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Indice '{index_name}' trovato, lo carico...")

# Carica l'indice Pinecone
pinecone_index = pc.Index(index_name)


vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index
        )

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")


index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
# Ora inserisci i nodi nel vector_store tramite index:
index.insert_nodes(nodes)
