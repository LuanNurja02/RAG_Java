import os
import csv
from dotenv import load_dotenv
from pinecone import Pinecone
import torch
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from rag import configure_query_engine
from util import (
    TUTOR_PROMPT, tutor
)

load_dotenv(dotenv_path='pinecone_key.env')

# Pinecone
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY non impostata")
pc = Pinecone(api_key=api_key)

# Costanti
TUTOR_INDEX_NAME = "documenti-extra"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_N = 3

# Modelli
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
reranker = FlagEmbeddingReranker(model=RERANKER_MODEL, top_n=RERANK_TOP_N)
llm = tutor  # Ollama LLM già configurato in util.py

# VectorStore
pinecone_index_tutor = pc.Index(TUTOR_INDEX_NAME)
vector_store_tutor = PineconeVectorStore(pinecone_index=pinecone_index_tutor)
storage_context_tutor = StorageContext.from_defaults(vector_store=vector_store_tutor)
vector_index_tutor = VectorStoreIndex.from_vector_store(
    vector_store=vector_store_tutor,
    embed_model=embed_model,
    storage_context=storage_context_tutor
)

# Query Engine
tutor_query_engine = configure_query_engine(
    index_instance=vector_index_tutor,
    llm_instance=llm,
    prompt_template_instance=TUTOR_PROMPT,
    reranker_instance=reranker,
    memory=None
)

# Evaluators
relevancy_evaluator = RelevancyEvaluator(llm=llm)
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)

# Lista di query da testare
queries = [
    "Mi spieghi il concetto di polimorfismo?",
    "Che cos'è l'ereditarietà in Java?",
    "A cosa serve il costrutto try-catch?",
    "Differenza tra interface e abstract class?",
    "Mi fai un esempio di classe esterna ed interna ",
    "Come si implementa una espressione lambda",
    "A cosa serve il costruttore?",
    "Come si fa l'overriding di un metodo?",
    "Come si leggono i file binari",
    "Mi spieghi la classe math, in particolare la funzione sin()?",
]

# CSV output
csv_filename = "evaluations.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["query", "retrieved_node", "relevancy_score", "relevancy_passing", "faithfulness_score", "faithfulness_passing"])

    for query in queries:
        print(f"\n🔎 Query: {query}")
        try:
            response = tutor_query_engine.query(query)
            response_str = response.response
            print(f"📝 Risposta generata:\n{response_str[:100]}...\n")

            # Valutazione per ogni nodo recuperato
            for i, source_node in enumerate(response.source_nodes, 1):
                context_text = source_node.get_content()
                print(f"📋 Valutazione nodo {i}/{len(response.source_nodes)}")

                # Relevancy evaluation per singolo nodo
                relevancy = relevancy_evaluator.evaluate(
                    query=query,
                    response=response_str,
                    contexts=[context_text],
                )

                # Faithfulness evaluation per singolo nodo
                faithfulness = faithfulness_evaluator.evaluate(
                    query=query,
                    response=response_str,
                    contexts=[context_text],
                )

                print(f"   ✓ Relevancy: {relevancy.passing} ({relevancy.score:.3f})")
                print(f"   ✓ Faithfulness: {faithfulness.passing} ({faithfulness.score:.3f})")

                # Salva nel CSV
                writer.writerow([
                    query,
                    context_text.replace("\n", " ").strip()[:100] + "..." if len(context_text) > 200 else context_text.replace("\n", " ").strip(),
                    f"{relevancy.score:.3f}",
                    relevancy.passing,
                    f"{faithfulness.score:.3f}",
                    faithfulness.passing
                ])

        except Exception as e:
            print(f"❌ Errore nella query '{query}': {str(e)}")
            # Registra anche gli errori nel CSV
            writer.writerow([query, f"ERRORE: {str(e)}", "N/A", "N/A", "N/A", "N/A"])

print(f"\n✅ Valutazione completata. Risultati salvati in: {csv_filename}")
print(f"📊 Totale query elaborate: {len(queries)}")