
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import SimpleDirectoryReader, Settings
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


from llama_index.core.node_parser import SentenceSplitter,SemanticSplitterNodeParser

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from transformers import AutoTokenizer, AutoModel


from llama_index.llms.ollama import Ollama
# PyMuPDF Reader example


def create_or_load_index():     
    # PyMuPDF Reader example     
    parser = PyMuPDFReader()     
    file_extractor = {         
        ".pdf": parser     
    }         
    
    # Carica documenti     
    documents = SimpleDirectoryReader(         
        "DATA/book", file_extractor=file_extractor     
    ).load_data()      
    
    print(f"Caricati {len(documents)} documenti")          
    
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2', padding_side='left')     
    embed_model = HuggingFaceEmbedding(                      
        model_name="intfloat/e5-small-v2"                           
    )      
    
    # Inizializza il parser per dividere i documenti in chunk     
    text_splitter = SentenceSplitter(     
        chunk_size=1000,      
        chunk_overlap=200       
    )            
    
    # Applica il chunking ai documenti         
    nodes = text_splitter.get_nodes_from_documents(documents)                
    print(f"Creati {len(nodes)} chunks")      
    
    # Aggiungi metadati personalizzati ai nodi     
    for i, node in enumerate(nodes):         
        # Estrai il nome del file dai metadati esistenti         
        file_name = node.metadata.get('file_name', 'unknown')         
        node.metadata.update({             
            'source_file': file_name,             
            'chunk_id': i,             
            'content_type': 'java_book'         
        })      
    
    # Set up Pinecone API key     
    os.environ["PINECONE_API_KEY"] = "pcsk_6VBs3G_8DQhyP34krGmTda5APDdDBnA849MLsswfmSukUhB4Ct1CaUsPvPWHGCVSuXdy5T"     
    api_key = os.environ["PINECONE_API_KEY"]      
    
    # Create Pinecone Vector Store     
    pc = Pinecone(api_key=api_key)     
    index_name = "libv3"      
    
    # Controlla se l'indice esiste già     
    if index_name not in pc.list_indexes():         
        print(f"Indice '{index_name}' non trovato, lo creo...")         
        pc.create_index(             
            name=index_name,             
            dimension=384,  # Dimensione dell'embedding model             
            metric="cosine",             
            spec=ServerlessSpec(                 
                cloud="aws",                  
                region="us-east-1"             
            ),         
        )         
        print("Indice creato, attendo inizializzazione...")         
        time.sleep(10)  # Attendi che l'indice sia pronto     
    else:         
        print(f"Indice '{index_name}' trovato")      
    
    # Carica l'indice Pinecone     
    pinecone_index = pc.Index(index_name)          
    
    # Controlla se l'indice è vuoto     
    index_stats = pinecone_index.describe_index_stats()     
    print(f"Statistiche indice PRIMA: {index_stats}")          
    
    # Crea vector store     
    vector_store = PineconeVectorStore(         
        pinecone_index=pinecone_index,     
    )      
    
    # Crea storage context     
    storage_context = StorageContext.from_defaults(vector_store=vector_store)      
    
    # SE L'INDICE È VUOTO, CREALO DAI NODES     
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
        print("Index caricato dall'indice esistente")
    
    return index   

def main():          
    index = create_or_load_index()  
    print(f"✅ Index creato/caricato con successo: {index}")
    return index      

if __name__ == "__main__":     
    main()