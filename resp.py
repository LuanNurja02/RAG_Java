import os
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


def main():
    print("📘 Java Tutor Chatbot (modalità terminale)")
    print("=" * 50)

    # Inizializzazione sistema
    try:
        print("🔧 Inizializzazione sistema...")

        # Configurazione Pinecone
        os.environ["PINECONE_API_KEY"] = "pcsk_6VBs3G_8DQhyP34krGmTda5APDdDBnA849MLsswfmSukUhB4Ct1CaUsPvPWHGCVSuXdy5T"
        api_key = os.environ["PINECONE_API_KEY"]

        # Inizializzazione embedding model
        embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-small-v2")

        # Connessione a Pinecone
        pc = Pinecone(api_key=api_key)
        index_name = "libv3"
        pinecone_index = pc.Index(index_name)

        # Configurazione vector store
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Creazione indice
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context
        )

        # Configurazione LLM ottimizzata per risposte dettagliate
        llm = Ollama(
            model="llama3.1:8b",
            temperature=0.3,  # Ridotta per risposte più precise
            max_tokens=2048,  # Aumentata per risposte più lunghe
            request_timeout=600,  # Aumentato per elaborazioni complesse
            context_window=8192,  # Aumentato per più contesto
            streaming=False  # Disabilitato per debug migliore
        )

        # Template del prompt migliorato per risposte dettagliate
        qa_prompt = PromptTemplate(
            """
Sei un tutor virtuale esperto di programmazione Java. Il tuo compito è fornire spiegazioni dettagliate e didattiche basate esclusivamente sul contesto fornito.

ISTRUZIONI IMPORTANTI:
1. Spiega SEMPRE il "perché" oltre al "come"
2. Fornisci esempi di codice Java quando possibile
3. Struttura la risposta in modo logico e progressivo
4. Se il contesto contiene codice, spiegalo riga per riga
5. Collega i concetti tra loro quando rilevante
6. Usa un linguaggio chiaro ma tecnico appropriato

Domanda dell'utente:
{query_str}

Contesto rilevante dalla documentazione:
---------------------
{context_str}
---------------------

RISPOSTA DETTAGLIATA:
Inizia la spiegazione partendo dai concetti fondamentali e sviluppa progressivamente verso aspetti più specifici. Includi sempre esempi pratici quando possibile.

"""
        )

        # Setup retriever con più documenti e filtro qualità
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,  # Ridotto per qualità migliore
            embed_model=embed_model
        )

        # Post-processor per filtrare documenti di bassa qualità
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.6  # Filtra documenti con score < 0.6
        )

        # Setup response synthesizer ottimizzato
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            streaming=False,
            response_mode="tree_summarize",  # Migliore per risposte dettagliate
            text_qa_template=qa_prompt,
            use_async=False
        )

        # Creazione Query Engine con post-processing
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[postprocessor]
        )

        print("✅ Sistema inizializzato con successo!")

    except Exception as e:
        print(f"❌ Errore durante l'inizializzazione: {str(e)}")
        return

    print("💡 Scrivi una domanda su Java (oppure 'esci' per terminare)")
    print("💡 Suggerimento: sii specifico per ottenere risposte più dettagliate")
    print("=" * 50)

    # Loop principale per le domande
    while True:
        try:
            user_input = input("\n👉 Domanda: ").strip()

            # Comandi di uscita
            if user_input.lower() in ["esci", "exit", "quit", "q"]:
                print("👋 Arrivederci!")
                break

            # Validazione input
            if not user_input:
                print("❓ Inserisci una domanda valida.")
                continue

            # Suggerimenti per domande troppo generiche
            if len(user_input.split()) < 3:
                print("💡 Suggerimento: prova a essere più specifico per ottenere una risposta migliore")

            print("🔍 Elaborazione in corso...")

            # Query al sistema RAG
            response = query_engine.query(user_input)

            # Debug: mostra i documenti utilizzati
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\n📦 Utilizzati {len(response.source_nodes)} documenti di riferimento:")
                for i, node in enumerate(response.source_nodes):
                    score = getattr(node, "score", 0)
                    content_preview = node.get_content() 
                    print(f"  [{i+1}] Score: {score:.3f}")
                    print(f"      Preview: {content_preview}")
                    print()

            # Mostra risposta
            print("\n📖 Risposta del Tutor:")
            print("=" * 60)
            
            response_text = str(response)
            if len(response_text.strip()) < 50:
                print("⚠️  Risposta troppo breve. Prova a riformulare la domanda o essere più specifico.")
            
            print(response_text)
            print("=" * 60)

            # Feedback sulla qualità
            if hasattr(response, 'source_nodes') and response.source_nodes:
                avg_score = sum(getattr(node, "score", 0) for node in response.source_nodes) / len(response.source_nodes)
                if avg_score < 0.7:
                    print("💡 La risposta potrebbe non essere completamente accurata. Prova a riformulare la domanda.")

        except KeyboardInterrupt:
            print("\n👋 Interruzione ricevuta. Arrivederci!")
            break
        except Exception as e:
            print(f"❌ Errore durante l'elaborazione: {str(e)}")
            print("Riprova con un'altra domanda.")


if __name__ == "__main__":
    main()