import os
import gradio as gr
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, get_response_synthesizer,StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever 
from llama_index.core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import torch

class JavaTutor:
    def __init__(self):
        """Inizializza il tutor Java con connessione a Pinecone e Ollama"""
        self.setup_connection()
        
    def setup_connection(self):
        """Configura la connessione a Pinecone e inizializza il query engine"""
        try:
            # 🔐 Set API Key (ricorda di non esporre mai in codice pubblico)
            os.environ["PINECONE_API_KEY"] = "pcsk_6VBs3G_8DQhyP34krGmTda5APDdDBnA849MLsswfmSukUhB4Ct1CaUsPvPWHGCVSuXdy5T"
            api_key = os.environ["PINECONE_API_KEY"]
            
            # 🔤 Embedding model
            self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
        
            
            # 🔍 Connessione a Pinecone
            pc = Pinecone(api_key=api_key)
            index_name = "java-api"
            pinecone_index = pc.Index(index_name)
            
            # 🧠 Ricostruzione vector store
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 🗃️ Costruzione indice
            self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=self.embed_model, storage_context=storage_context)
            
            # 🤖 LLM tramite Ollama
            self.llm = Ollama(
                            model="llama3",
                            temperature=0.4,
                            max_tokens=512, 
                            request_timeout=200, 
                            context_window=-1,
                            streaming=True 
                        )
            
            # 📄 Prompt contestualizzato (didattico Java)
            context_template = """\
            Sei un tutor (italiano) esperto di Java. Il tuo obiettivo è insegnare concetti di programmazione Java in modo semplice e didattico.
            {query}
            Usa le informazioni fornite nel contesto qui sotto per spiegare all'utente il concetto richiesto.
            
            non aggiungere informazioni oltre al contesto che ti ho fornito\
            ---------------------
            {context_str}
            ---------------------
            Rispondi come se fossi un insegnante, fornendo esempi di codice, definizioni chiare, e spiegazioni passo-passo.
            Se non hai abbastanza informazioni, rispondi "Non ho abbastanza informazioni per rispondere a questa domanda.".
            """
            
            text_qa_template = PromptTemplate(context_template)
            
            # 🔍 Retriever + ✍️ Synthesizer
            retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3, embed_model=self.embed_model)
            response_synthesizer = get_response_synthesizer(llm=self.llm, text_qa_template=text_qa_template)
            
            # 🎯 Query Engine
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )
            
            self.is_initialized = True
            
        except Exception as e:
            self.is_initialized = False
            self.error_message = f"Errore durante l'inizializzazione: {str(e)}"
    
    def ask_question(self, question):
        """Processa una domanda dell'utente e restituisce la risposta"""
        if not self.is_initialized:
            return f"❌ Sistema non inizializzato: {self.error_message}"
        
        if not question.strip():
            return "❓ Per favore, inserisci una domanda!"
        
        try:
            # ▶️ Esegui la query
            response = self.query_engine.query(question)
            
            # 🎨 Formatta la risposta
            formatted_response = f"""
📘 **Risposta del Tutor Java:**

{response.response}

---
💡 *Hai altre domande su Java? Chiedi pure!*
            """
            
            return formatted_response
            
        except Exception as e:
            return f"❌ Errore durante la ricerca: {str(e)}"

def create_interface():
    """Crea l'interfaccia Gradio"""
    
    # Inizializza il tutor
    tutor = JavaTutor()
    
    # Esempi di domande predefinite
    examples = [
        "Cosa sono le liste in Java?",
        "Come funzionano i cicli for?",
        "Cosa sono le classi e gli oggetti?",
        "Differenza tra ArrayList e LinkedList",
        "Come gestire le eccezioni in Java?",
        "Cosa sono i metodi statici?",
        "Ereditarietà in Java",
        "Cosa sono le interfacce?"
    ]
    
    # Crea l'interfaccia
    with gr.Blocks(
        title="☕ Java Tutor - RAG Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px !important;
            margin: auto !important;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="title">
            <h1>☕ Java Tutor - Assistente RAG</h1>
        </div>
        <div class="subtitle">
            <p>Il tuo tutor personale per imparare Java! Fai domande sui concetti di programmazione Java.</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="🤔 Fai una domanda su Java",
                    placeholder="Es: Cosa sono le liste in Java?",
                    lines=2,
                    max_lines=4
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Chiedi al Tutor", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 Pulisci", scale=1)
        
        with gr.Row():
            response_output = gr.Markdown(
                label="📖 Risposta del Tutor",
                value="👋 Ciao! Sono il tuo tutor Java. Fai pure la tua prima domanda!",
                elem_id="response_area"
            )
        
        with gr.Row():
            gr.Examples(
                examples=examples,
                inputs=question_input,
                label="💡 Esempi di domande che puoi fare:"
            )
        
        # Status del sistema
        with gr.Row():
            status_text = gr.Textbox(
                label="📊 Status Sistema",
                value="🟢 Sistema pronto" if tutor.is_initialized else f"🔴 Errore: {tutor.error_message}",
                interactive=False,
                max_lines=1
            )
        
        # Eventi
        submit_btn.click(
            fn=tutor.ask_question,
            inputs=[question_input],
            outputs=[response_output]
        )
        
        question_input.submit(
            fn=tutor.ask_question,
            inputs=[question_input],
            outputs=[response_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "👋 Ciao! Sono il tuo tutor Java. Fai pure la tua prima domanda!"),
            outputs=[question_input, response_output]
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
            <p>🔧 Powered by LlamaIndex + Pinecone + Ollama</p>
        </div>
        """)
    
    return demo

def main():
    """Funzione principale per lanciare l'applicazione"""
    print("🚀 Avvio Java Tutor RAG Interface...")
    
    # Crea l'interfaccia
    demo = create_interface()
    
    # Lancia l'applicazione
    demo.launch(
        server_name="0.0.0.0",  # Accessibile da tutti gli indirizzi
        server_port=7860,       # Porta predefinita
        share=True,            # Cambia a True per condividere pubblicamente
        show_api=False,         # Nasconde l'API docs
        show_error=True,        # Mostra errori dettagliati
        debug=False             # Modalità debug
    )

if __name__ == "__main__":
    main()