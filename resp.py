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
import torch
def main():
    print("Java Tutor Chatbot")
    print("=" * 50)

    # Inizializzazione sistema
    try:
        torch.cuda.empty_cache()
        print("🔧 Inizializzazione sistema...")

        # Configurazione Pinecone
        os.environ["PINECONE_API_KEY"] = "pcsk_6VBs3G_8DQhyP34krGmTda5APDdDBnA849MLsswfmSukUhB4Ct1CaUsPvPWHGCVSuXdy5T"
        api_key = os.environ["PINECONE_API_KEY"]

        # Inizializzazione embedding model
        embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-small-v2")

        # Connessione a Pinecone
        pc = Pinecone(api_key=api_key)
        index_name = "meta-lib"
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

    
        qa_prompt = PromptTemplate(
            """Sei un tutor esperto di programmazione Java. Devi fornire una spiegazione dettagliata e didattica basata ESCLUSIVAMENTE sulle informazioni fornite nel contesto.
                parla quindi delle informazioni recuperate e rispondi alla domanda
    -------------------------------------------
                {context_str}
    ------------------------------------------.
                query: {query_str}
                
                RISPOSTA DETTAGLIATA:"""
        )


    
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=3,  # Ridotto per qualità migliore
            embed_model=embed_model,
            sparse_top_k=2
            

        )

        # Post-processor per filtrare documenti di bassa qualità
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.80 # Filtra documenti con score < 0.6
        )

    
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            streaming=False,
            response_mode="refine",  # Migliore per risposte dettagliate
            text_qa_template=qa_prompt,
            use_async=False
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[postprocessor]
        )

        print(" Sistema inizializzato con successo!")

    except Exception as e:
        print(f" Errore durante l'inizializzazione: {str(e)}")
        return

    print(" Scrivi una domanda su Java (oppure 'esci' per terminare)")
    print(" Suggerimento: sii specifico per ottenere risposte più dettagliate")
    print("=" * 50)


    while True:
        try:
            user_input = input("\n Domanda: ").strip()

            # Comandi di uscita
            if user_input.lower() in ["esci", "exit", "quit", "q"]:
                print(" Arrivederci!")
                break

            # Validazione input
            if not user_input:
                print("❓ Inserisci una domanda valida.")
                continue

            # Suggerimenti per domande troppo generiche
            if len(user_input.split()) < 3:
                print("💡 Suggerimento: prova a essere più specifico per ottenere una risposta migliore")

            print("🔍 Elaborazione in corso...")

            # Query
            response = query_engine.query(user_input)

            #mostra i documenti utilizzati
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\n Utilizzati {len(response.source_nodes)} documenti di riferimento:")
                for i, node in enumerate(response.source_nodes):
                    score = getattr(node, "score", 0)
                    content_preview = node.get_content() 
                    print(f"  [{i+1}] Score: {score:.3f}")
                    print(f"      Preview: {content_preview}")
                    print()

            print("\n📖 Risposta del Tutor:")
            print("=" * 60)
            
            response_text = str(response)
            if len(response_text.strip()) < 50:
                print("Risposta troppo breve.")
            
            print(response_text)
            print("=" * 60)

            # Feedback sulla qualità
            if hasattr(response, 'source_nodes') and response.source_nodes:
                avg_score = sum(getattr(node, "score", 0) for node in response.source_nodes) / len(response.source_nodes)
                if avg_score < 0.7:
                    print("La risposta potrebbe non essere completamente accurata. Prova a riformulare la domanda.")

        except KeyboardInterrupt:
            print("\nInterruzione ricevuta. Arrivederci!")
            break
        except Exception as e:
            print(f"Errore durante l'elaborazione: {str(e)}")
            print("Riprova con un'altra domanda.")


if __name__ == "__main__":
    main()