import os
import time
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import torch
import gradio as gr
from llama_index.core.response_synthesizers import ResponseMode
from gradio import themes
from dotenv import load_dotenv

# Importa per il re-ranker
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


# Importa le configurazioni da util.py
from util import (
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_REQUEST_TIMEOUT,
    TUTOR_PROMPT,
    SPIEGAZIONE_CODICE_PROMPT,
    DEBUG_CODICE_PROMPT,
    CREA_CODICE_PROMPT
)

load_dotenv()

#indici Pinecone
TUTOR_INDEX_NAME = "meta-lib"
ASSISTANT_INDEX = "java-codebase"

# Configurazione del modello di embedding
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
EMBEDDING_DIMENSION = 384

#Re-ranker
RERANK_MODEL_NAME = "BAAI/bge-reranker-base" 
RERANK_TOP_N = 2 #nodi da usare dopo il rerenking

# Funzione per Configurare il Query Engine 
def configure_query_engine(index_instance, llm_instance, embed_model_instance, prompt_template_instance, reranker_instance):
    retriever = VectorIndexRetriever(
        index=index_instance,
        similarity_top_k=5,
        embed_model=embed_model_instance,
        sparse_top_k=2
    )

    node_postprocessors = [
        # SimilarityPostprocessor(similarity_cutoff=0.80),
        reranker_instance
    ]

    response_synthesizer = get_response_synthesizer(
        llm=llm_instance,
        streaming=False,
        response_mode=ResponseMode.COMPACT,
        text_qa_template=prompt_template_instance,
        use_async=False
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors
    )


llm = None
embed_model = None
reranker = None 
vector_indices = {}
query_engines = {}

try:
    torch.cuda.empty_cache()
    print("🔧 Inizializzazione globale LLM, embedding, re-ranker e indici Pinecone...")

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("La variabile d'ambiente 'PINECONE_API_KEY' non è impostata.")

    pc = Pinecone(api_key=api_key)

    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Modello di embedding caricato.")

    # Inizializzazione del re-ranker
    print(f"🔄 Caricamento modello di re-ranking '{RERANK_MODEL_NAME}'...")
    
    reranker = FlagEmbeddingReranker(
        model=RERANK_MODEL_NAME,
        top_n=RERANK_TOP_N 
    
    )
    print("✅ Modello di re-ranking caricato.")


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
    print(f"🔄 Caricamento indice '{ASSISTANT_INDEX}' per la modalità Coding Assistant...")
    pinecone_index_coding_assistant = pc.Index(ASSISTANT_INDEX)
    vector_store_coding_assistant = PineconeVectorStore(pinecone_index=pinecone_index_coding_assistant)
    storage_context_coding_assistant = StorageContext.from_defaults(vector_store=vector_store_coding_assistant)
    vector_indices["Coding Assistant"] = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_coding_assistant,
        embed_model=embed_model,
        storage_context=storage_context_coding_assistant
    )
    print(f"✅ Indice '{ASSISTANT_INDEX}' caricato.")

    # Prepara il query engine di default per la modalità Tutor
    query_engines["Tutor"] = configure_query_engine(
        index_instance=vector_indices["Tutor"],
        llm_instance=llm,
        embed_model_instance=embed_model,
        prompt_template_instance=TUTOR_PROMPT,
        reranker_instance=reranker
    )
    print("✅ Query engine per 'tutor' pronto.")

except Exception as e:
    print(f"❌ Errore critico durante l'inizializzazione globale: {str(e)}")
    raise


def gradio_rag_interface(mode, domanda, codice, prompt_mode):

    codice = codice if codice is not None else ""

    if codice.strip():
        full_query = f"{domanda}\n\nCODICE FORNITO:\n```java\n{codice.strip()}\n```"
    else:
        full_query = domanda

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
                selected_prompt_template = SPIEGAZIONE_CODICE_PROMPT #DEFAULT

            current_query_engine = configure_query_engine(
                index_instance=vector_indices["Coding Assistant"],
                llm_instance=llm,
                embed_model_instance=embed_model,
                prompt_template_instance=selected_prompt_template,
                reranker_instance=reranker
            )
            print(f"💡 Esecuzione in modalità CODING ASSISTANT ({prompt_mode}) con domanda: {domanda[:50]}...")

        response = current_query_engine.query(full_query)

        sources_text = ""
        if hasattr(response, 'source_nodes') and response.source_nodes:
            sources_text += "### Documenti di Riferimento Utilizzati\n"
            for i, node in enumerate(response.source_nodes):
                score = getattr(node, "score", None)
                content_preview = node.get_content()
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "...\n(Contenuto troncato)"
                score_str = f"Score: {score:.3f}" if score is not None else "Score: N/A"
                sources_text += f"**[{i+1}] {score_str}**\n```\n{content_preview}\n```\n\n"

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
with gr.Blocks(theme=themes.Soft()) as demo:
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
                visible=False,
                interactive=True
            )


            with gr.Row():
                btn_submit = gr.Button("Invia ", variant="primary", icon="DATA/gui_icon/coffee.png")
                btn_clear = gr.Button("Cancella ", variant="secondary", icon="DATA/gui_icon/trash.png")

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
                visible=False
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            risposta = gr.Markdown(label="Risposta dell'Assistente", value="Risposta generata dal modello apparirà qui.")
        with gr.Column():
            fonti = gr.Markdown(label="Documenti di Riferimento Utilizzati", value="Le fonti recuperate appariranno qui.")

    def update_ui_visibility(selected_mode):
        if selected_mode == "Coding Assistant":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    mode.change(
        fn=update_ui_visibility,
        inputs=mode,
        outputs=[prompt_mode, codice],
        queue=False
    )

    btn_submit.click(
        fn=gradio_rag_interface,
        inputs=[mode, domanda, codice, prompt_mode],
        outputs=[risposta, fonti]
    )


    btn_clear.click(
        fn=lambda: (
            "",
            "",
            "Risposta generata dal modello apparirà qui.",
            "Le fonti recuperate appariranno qui.",
            gr.update(value="Spiegazione", visible=False),
            gr.update(value="Tutor")
        ),
        outputs=[domanda, codice, risposta, fonti, prompt_mode, mode],
        queue=False
    )

demo.launch()