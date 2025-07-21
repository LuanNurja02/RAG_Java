import os
import time
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
import gradio as gr
from llama_index.core.response_synthesizers import ResponseMode
from gradio import themes



# Nomi degli indici Pinecone
TUTOR_INDEX_NAME = "meta-lib" 
CODING_ASSISTANT_INDEX_NAME = "java-codebase" 

# Configurazione del modello di embedding
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
EMBEDDING_DIMENSION = 384 

# Parametri del modello LLM (Ollama)
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_TOKENS = 14000
OLLAMA_CONTEXT_WINDOW = 14000
OLLAMA_REQUEST_TIMEOUT = 600

# --- Prompt per le Diverse Modalità ---

TUTOR_PROMPT = PromptTemplate(
    """Sei un tutor esperto di programmazione Java. Devi fornire una spiegazione dettagliata e didattica basata ESCLUSIVAMENTE sulle informazioni fornite nel contesto.
    Parla quindi delle informazioni recuperate e rispondi alla domanda.
-------------------------------------------
{context_str}
------------------------------------------.
query: {query_str}

RISPOSTA DETTAGLIATA:"""
)

SPIEGAZIONE_CODICE_PROMPT = PromptTemplate(
    """Sei un assistente di programmazione Java. Il tuo compito è spiegare dettagliatamente il frammento di codice o la funzionalità a cui la query si riferisce, basandoti sul contesto fornito.
    Spiega la logica, le classi, i metodi e le interazioni.
-------------------------------------------
{context_str}
------------------------------------------.
query: {query_str}

SPIEGAZIONE DETTAGLIATA DEL CODICE:"""
)

DEBUG_CODICE_PROMPT = PromptTemplate(
    """Sei un assistente di debugging Java. Analizza il frammento di codice o il problema descritto nella query, basandoti sul contesto fornito.
    Identifica potenziali bug, errori logici o inefficienze e suggerisci soluzioni concrete ed esempi, se possibile.
-------------------------------------------
{context_str}
------------------------------------------.
query: {query_str}

ANALISI E SUGGERIMENTI PER IL DEBUG:"""
)

CREA_CODICE_PROMPT = PromptTemplate(
    """Sei un assistente di generazione codice Java. Basandoti sul contesto fornito e sulla richiesta nella query, genera un frammento di codice Java funzionale.
    Fornisci solo il codice necessario e, se utile, un breve commento sulla logica.
    NON includere spiegazioni extra o testo che non sia codice.
-------------------------------------------
{context_str}
------------------------------------------.
query: {query_str}

CODICE JAVA GENERATO:"""
)

# --- Funzione per Configurare il Query Engine (riutilizzabile) ---
def configure_query_engine(index_instance, llm_instance, embed_model_instance, prompt_template_instance):
    retriever = VectorIndexRetriever(
        index=index_instance,
        similarity_top_k=3,
        embed_model=embed_model_instance,
        sparse_top_k=2
    )

    postprocessor = SimilarityPostprocessor(
        similarity_cutoff=0.80
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm_instance,
        streaming=False,
        response_mode=ResponseMode.REFINE,
        text_qa_template=prompt_template_instance,
        use_async=False
    )

    return RetrieverQueryEngine( # Modifica suggerita da Sourcery per concisione
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[postprocessor]
    )


llm = None
embed_model = None
vector_indices = {}
query_engines = {}

try:
    torch.cuda.empty_cache()
    print("🔧 Inizializzazione globale LLM, embedding e indici Pinecone...")

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("La variabile d'ambiente 'PINECONE_API_KEY' non è impostata.")

    pc = Pinecone(api_key=api_key)

    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, device="cuda")
    print("✅ Modello di embedding caricato.")

    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        max_tokens=OLLAMA_MAX_TOKENS,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
        context_window=OLLAMA_CONTEXT_WINDOW,
        streaming=False,
        min_length=100,
        top_p=0.9,
        repeat_penalty=1.2
    )
    print("✅ Modello LLM Ollama inizializzato.")

    # Indice per la modalità Tutor
    print(f"🔄 Caricamento indice '{TUTOR_INDEX_NAME}' per la modalità Tutor...")
    pinecone_index_tutor = pc.Index(TUTOR_INDEX_NAME)
    vector_store_tutor = PineconeVectorStore(pinecone_index=pinecone_index_tutor)
    storage_context_tutor = StorageContext.from_defaults(vector_store=vector_store_tutor)
    vector_indices["Tutor"] = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_tutor,
        embed_model=embed_model,
        storage_context=storage_context_tutor
    )
    print(f"✅ Indice '{TUTOR_INDEX_NAME}' caricato.")

    # Indice per la modalità Coding Assistant
    print(f"🔄 Caricamento indice '{CODING_ASSISTANT_INDEX_NAME}' per la modalità Coding Assistant...")
    pinecone_index_coding_assistant = pc.Index(CODING_ASSISTANT_INDEX_NAME)
    vector_store_coding_assistant = PineconeVectorStore(pinecone_index=pinecone_index_coding_assistant)
    storage_context_coding_assistant = StorageContext.from_defaults(vector_store=vector_store_coding_assistant)
    vector_indices["Coding Assistant"] = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_coding_assistant,
        embed_model=embed_model,
        storage_context=storage_context_coding_assistant
    )
    print(f"✅ Indice '{CODING_ASSISTANT_INDEX_NAME}' caricato.")

    # Prepara il query engine di default per la modalità Tutor
    query_engines["Tutor"] = configure_query_engine(
        index_instance=vector_indices["Tutor"],
        llm_instance=llm,
        embed_model_instance=embed_model,
        prompt_template_instance=TUTOR_PROMPT
    )
    print("✅ Query engine per 'tutor' pronto.")

except Exception as e:
    print(f"❌ Errore critico durante l'inizializzazione globale: {str(e)}")
    raise


def gradio_rag_interface(mode, domanda, codice, prompt_mode):
    
    codice = codice if codice is not None else ""
    
    # Prepara la query completa per l'LLM
    if codice.strip():
        full_query = f"{domanda}\n\nCODICE FORNITO:\n```java\n{codice.strip()}\n```"
    else:
        full_query = domanda

    # Validazione input
    if not full_query.strip():
        return "❓ Per favore, inserisci almeno una domanda o del codice da analizzare.", ""

    try:
        if mode == "Tutor":
            current_query_engine = query_engines["Tutor"]
            print(f"💡 Esecuzione in modalità TUTOR con domanda: {domanda[:50]}...")
            
        else: # mode == "coding_assistant"
            if prompt_mode == "Spiegazione":
                selected_prompt_template = SPIEGAZIONE_CODICE_PROMPT
            elif prompt_mode == "Debug":
                selected_prompt_template = DEBUG_CODICE_PROMPT
            elif prompt_mode == "Crea":
                selected_prompt_template = CREA_CODICE_PROMPT
            else:
                selected_prompt_template = SPIEGAZIONE_CODICE_PROMPT
            
            current_query_engine = configure_query_engine(
                index_instance=vector_indices["Coding Assistant"],
                llm_instance=llm,
                embed_model_instance=embed_model,
                prompt_template_instance=selected_prompt_template
            )
            print(f"💡 Esecuzione in modalità CODING ASSISTANT ({prompt_mode}) con domanda: {domanda[:50]}...")

        # Esegui la query
        response = current_query_engine.query(full_query)

        # Prepara la stringa delle fonti per la visualizzazione in Gradio
        sources_text = ""
        if hasattr(response, 'source_nodes') and response.source_nodes:
            sources_text += "### Documenti di Riferimento Utilizzati\n"
            for i, node in enumerate(response.source_nodes):
                score = getattr(node, "score", 0)
                content_preview = node.get_content()
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "...\n(Contenuto troncato)"
                sources_text += f"**[{i+1}] Score: {score:.3f}**\n```\n{content_preview}\n```\n\n"
        
        response_content = str(response)
        if not response_content.strip():
            response_content = "Non ho trovato una risposta rilevante basandomi sulle informazioni disponibili."
        elif len(response_content.strip()) < 50 and not sources_text:
            response_content += "\n\n💡 La risposta potrebbe essere troppo breve. Prova a riformulare la domanda o a fornire più contesto."

        return response_content, sources_text

    except Exception as e:
        error_message = f"Si è verificato un errore: {str(e)}\nPer favore, riprova."
        print(f"❌ Errore durante l'elaborazione della query: {str(e)}")
        return error_message, ""


# Interfaccia Gradio 
with gr.Blocks(theme=themes.Soft()) as demo: # Tema Soft per un aspetto più moderno
    gr.Markdown("# ☕Java AI Assistant")
    gr.Markdown(
        "Benvenuto! Sono il tuo assistente per la programmazione Java. "
        "Scegli una **modalità** qui sotto e poi scrivi la tua domanda."
        " Nella modalità **Coding Assistant**, puoi anche incollare del codice Java."
    )

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Tutor", "Coding Assistant"],
                value="Tutor",
                label="Seleziona la Modalità",
                info="**Tutor:** concetti Java e documentazione. **Coding Assistant:** assistenza su codice specifico",
                interactive=True
            )
            prompt_mode = gr.Radio(
                ["Spiegazione", "Debug", "Crea"],
                value="Spiegazione",
                label="Tipo di Assistenza Codice",
                info="Scegli il tipo di assistenza per il Coding Assistant.",
                visible=False, # Inizialmente nascosto
                interactive=True
            )
            
            
            with gr.Row():
                btn_submit = gr.Button("Invia ", variant="primary", icon="DATA/gui_icon/coffee.png")
                btn_clear = gr.Button("Cancella ", variant="secondary", icon="DATA/gui_icon/trash.png") # Pulsante per cancellare

        with gr.Column(scale=2):
            domanda = gr.Textbox(
                label="La tua Domanda",
                lines=3,
                placeholder="Es: 'Spiega l'ereditarietà in Java' o 'Cosa fa questo blocco di codice?'",
                interactive=True
            )
            codice = gr.Textbox(
                label="Codice Java (Opzionale per Coding Assistant)",
                lines=7,
                placeholder="Incolla qui il codice Java da analizzare, debuggare o su cui vuoi basare una generazione. (Ignorato in modalità Tutor)",
                interactive=True,
                visible=False # Inizialmente nascosto
            )
    
    gr.Markdown("---") 
    
    with gr.Row():
        with gr.Column():
            risposta = gr.Markdown(label="Risposta dell'Assistente", value="Risposta generata dal modello apparirà qui.")
        with gr.Column():
            fonti = gr.Markdown(label="Documenti di Riferimento Utilizzati", value="Le fonti recuperate appariranno qui.")

    # Logica per mostrare/nascondere il selettore del tipo di prompt e il campo codice
    def update_ui_visibility(selected_mode):
        if selected_mode == "Coding Assistant":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    mode.change(
        fn=update_ui_visibility,
        inputs=mode,
        outputs=[prompt_mode, codice],
        queue=False # Evita di mettere in coda se l'utente cambia rapidamente la modalità
    )
    
    btn_submit.click(
        fn=gradio_rag_interface,
        inputs=[mode, domanda, codice, prompt_mode],
        outputs=[risposta, fonti]
    )


    btn_clear.click(
        fn=lambda: (
            "", # domanda
            "", # codice
            "Risposta generata dal modello apparirà qui.", # risposta
            "Le fonti recuperate appariranno qui.", # fonti
            gr.update(value="Spiegazione", visible=False), # prompt_mode
            gr.update(value="Tutor") # mode
        ),
        outputs=[domanda, codice, risposta, fonti, prompt_mode, mode],
        queue=False 
    )

demo.launch()