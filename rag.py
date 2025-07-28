import os
from dotenv import load_dotenv
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
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
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
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
load_dotenv(dotenv_path='pinecone_key.env')


TUTOR_INDEX_NAME = "meta-lib"
ASSISTANT_INDEX = "java-codebase"
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
# Re-ranker
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
RERANK_TOP_N = 3  # nodi da usare dopo il re-ranking

# Funzione per Configurare il Query Engine
def configure_query_engine(index_instance, llm_instance, prompt_template_instance, reranker_instance, response_mode=ResponseMode.COMPACT, memory=None):
    retriever = VectorIndexRetriever(
        index=index_instance,
        similarity_top_k=10
    )

    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.75),
        reranker_instance
    ]

    response_synthesizer = get_response_synthesizer(
        llm=llm_instance,
        streaming=True,
        response_mode=response_mode,
        text_qa_template=prompt_template_instance
    )

    if memory is not None:
        # Modalità chat se è impostata la memoria 
        return ContextChatEngine(
            retriever=retriever,
            llm=llm_instance,
            memory=memory,
            node_postprocessors=node_postprocessors,
            prefix_messages=[]
        )
    else:
        # altrimenti modalità classca q&a
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors
        )

# funzione di mapping per la modalità di risposta
RESPONSE_MODE_MAP = {
    "Dettagliata": ResponseMode.COMPACT,
    "Sintetica": ResponseMode.TREE_SUMMARIZE
}

llm_tutor = None
llm_coding = None
embed_model = None
reranker = None
vector_indices = {}

try:
    torch.cuda.empty_cache()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("The 'PINECONE_API_KEY' environment variable is not set.")

    pc = Pinecone(api_key=api_key)

    # Embedding
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, device=embed_device)

    #re-ranker
    reranker = FlagEmbeddingReranker(
        model=RERANK_MODEL_NAME,
        top_n=RERANK_TOP_N
    )

    # LLM per Tutor (generalista)
    llm_tutor = Ollama(
        model=OLLAMA_MODEL,  # es: "llama3.1:8b"
        temperature=OLLAMA_TEMPERATURE,
        max_tokens=OLLAMA_MAX_TOKENS,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
        context_window=OLLAMA_CONTEXT_WINDOW,
        streaming=True,
        min_length=100,
        top_p=0.9,
        repeat_penalty=1.2
    )

    # LLM per Coding Assistant (esperto in codice)
    llm_coding = Ollama(
        model="codellama:7b",  # Modello più leggero che richiede meno memoria
        temperature=0.1,
        max_tokens=OLLAMA_MAX_TOKENS,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
        context_window=OLLAMA_CONTEXT_WINDOW,
        streaming=True,
        min_length=100,
        top_p=0.9,
        repeat_penalty=1.2
    )

    pinecone_index_tutor = pc.Index(TUTOR_INDEX_NAME)
    vector_store_tutor = PineconeVectorStore(pinecone_index=pinecone_index_tutor)
    storage_context_tutor = StorageContext.from_defaults(vector_store=vector_store_tutor)
    vector_indices["Tutor"] = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_tutor,
        embed_model=embed_model,
        storage_context=storage_context_tutor
    )

    pinecone_index_coding_assistant = pc.Index(ASSISTANT_INDEX)
    vector_store_coding_assistant = PineconeVectorStore(pinecone_index=pinecone_index_coding_assistant)
    storage_context_coding_assistant = StorageContext.from_defaults(vector_store=vector_store_coding_assistant)
    vector_indices["Coding Assistant"] = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_coding_assistant,
        embed_model=embed_model,
        storage_context=storage_context_coding_assistant
    )

    # Inizializza le memorie globali per Tutor e Coding Assistant
    chat_memory_tutor = ChatMemoryBuffer.from_defaults(token_limit=5000)
    chat_memory_coding = ChatMemoryBuffer.from_defaults(token_limit=5000)

except Exception as e:
    print(f"Critical error during global initialization: {str(e)}")
    raise


def process_message(message: str, history: list, mode: str, prompt_mode: str, codice: str, response_mode_tutor: str, chat_mode: str):

    history.append([message, None])

    final_query = message 

    if codice and codice.strip():
        full_query = f"{final_query}\n\nCODICE FORNITO:\n```java\n{codice.strip()}\n```"
    else:
        full_query = final_query

    if not full_query.strip():
        if (
            history
            and history[-1][0] == final_query
            and history[-1][1] is None
        ):
            history.pop()
        yield history, "Please enter at least a question or code to analyze."
        return

    current_response_text = ""

    try:
        if mode == "Tutor":
            selected_response_mode = RESPONSE_MODE_MAP.get(response_mode_tutor, ResponseMode.COMPACT) if chat_mode == "Classica" else ResponseMode.COMPACT

            if chat_mode == "Chat":
                current_query_engine = configure_query_engine(

                    index_instance=vector_indices["Tutor"],
                    llm_instance=llm_tutor,  # Usa LLM Tutor
                    prompt_template_instance=TUTOR_PROMPT,
                    reranker_instance=reranker,
                    response_mode=ResponseMode.COMPACT,
                    memory=chat_memory_tutor

                )
                print(f"💡 Executing in TUTOR mode (Chat) with query: {final_query[:50]}...")
                streaming_response = current_query_engine.stream_chat(full_query)
            else:
                current_query_engine = configure_query_engine(

                    index_instance=vector_indices["Tutor"],
                    llm_instance=llm_tutor,  # Usa LLM Tutor
                    prompt_template_instance=TUTOR_PROMPT,
                    reranker_instance=reranker,
                    response_mode=selected_response_mode,
                    memory=None

                )
                print(
                    f"💡 Executing in TUTOR mode (Classica) with query: {final_query[:50]}... (ResponseMode: {response_mode_tutor})"
                )
                streaming_response = current_query_engine.query(full_query)
        else: # mode == "Coding Assistant"
            if prompt_mode == "Crea":
                selected_prompt_template = CREA_CODICE_PROMPT
            elif prompt_mode == "Debug":
                selected_prompt_template = DEBUG_CODICE_PROMPT
            else:
                selected_prompt_template = SPIEGAZIONE_CODICE_PROMPT # DEFAULT for Coding Assistant

            if chat_mode == "Chat":
                current_query_engine = configure_query_engine(

                    index_instance=vector_indices["Coding Assistant"],
                    llm_instance=llm_coding,  # Usa LLM Coding Assistant
                    prompt_template_instance=selected_prompt_template,
                    reranker_instance=reranker,
                    memory=chat_memory_coding

                )
                print(
                    f"💡 Executing in CODING ASSISTANT mode (Chat) ({prompt_mode}) with query: {final_query[:50]}..."
                )
                streaming_response = current_query_engine.stream_chat(full_query)
            else:
                current_query_engine = configure_query_engine(

                    index_instance=vector_indices["Coding Assistant"],
                    llm_instance=llm_coding,  # Usa LLM Coding Assistant
                    prompt_template_instance=selected_prompt_template,
                    reranker_instance=reranker,
                    memory=None

                )
                print(
                    f"💡 Executing in CODING ASSISTANT mode (Classica) ({prompt_mode}) with query: {final_query[:50]}..."
                )
                streaming_response = current_query_engine.query(full_query)


        response_index = -1 

        if hasattr(streaming_response, 'response_gen'):
            for text_chunk in streaming_response.response_gen:
                current_response_text += text_chunk
                history[response_index][1] = current_response_text
                yield history, ""
        else:
            current_response_text = str(getattr(streaming_response, 'response', streaming_response))
            history[response_index][1] = current_response_text
            yield history, ""

        sources_output_text = ""
        if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
            sources_output_text += "### Documenti di Riferimento Utilizzati\n"
            for i, node in enumerate(streaming_response.source_nodes):
                score = getattr(node, "score", None)
                content_preview = node.get_content()
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "...\n(Contenuto troncato)"
                score_str = f"Score: {score:.3f}" if score is not None else "Score: N/A"
                sources_output_text += f"**[{i+1}] {score_str}**\n```\n{content_preview}\n```\n\n"

        yield history, sources_output_text

    except Exception as e:
        error_message = f"Si è verificato un errore: {str(e)}\nPer favore riprova."
        print(f"Error processing query: {str(e)}")
        history[-1][1] = error_message
        yield history, ""


# Gradio Interface
with gr.Blocks(theme=themes.Ocean(), title="Java Assistant") as demo:
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
            chat_mode = gr.Radio(
                ["Classica", "Chat"],
                value="Classica",
                label="Modalità conversazione",
                info="Classica: risposta singola, Chat: memoria conversazionale",
                visible=True,
                interactive=True
            )

            response_mode_tutor = gr.Radio(
                ["Dettagliata", "Sintetica"],
                value="Dettagliata",
                label="Modalità di risposta",
                info="<b>Dettagliata</b>: risposta completa e approfondita. <b>Sintetica</b>: risposta riassuntiva e strutturata ad albero, utile per panoramiche rapide.",
                visible=True, # Initially visible if Classica is default
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
            codice = gr.Textbox(
                label="Codice Java (Opzionale per Coding Assistant)",
                lines=7,
                placeholder="Incolla qui il codice Java da analizzare, debuggare o su cui vuoi basare una generazione. (Ignorato in modalità Tutor)",
                interactive=True,
                visible=False
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversazione",
                elem_id="chatbot",
                height=500
            )
            fonzi = gr.Markdown(label="Documenti di Riferimento Utilizzati", value="Le fonti recuperate appariranno qui.")

            with gr.Row():
                domanda_input = gr.Textbox(
                    label="Il tuo messaggio",
                    lines=2,
                    placeholder="Digita la tua domanda qui...",
                    interactive=True,
                    scale=8
                )

                with gr.Column(scale=2, min_width=100):
                    btn_submit = gr.Button("Invia", variant="primary", icon="DATA/gui_icon/coffee.png")
                    btn_clear = gr.Button("Cancella", variant="secondary", icon="DATA/gui_icon/trash.png")

    mode.change(
        fn=lambda selected_mode: (
            gr.update(visible=selected_mode == "Coding Assistant"), # prompt_mode
            gr.update(visible=selected_mode == "Coding Assistant"), # codice
            gr.update(visible=selected_mode == "Tutor") # response_mode_tutor (initial state based on mode)
        ),
        inputs=mode,
        outputs=[prompt_mode, codice, response_mode_tutor],
        queue=False
    )

    # New event listener for chat_mode to control response_mode_tutor visibility
    chat_mode.change(
        fn=lambda selected_chat_mode, current_mode: gr.update(visible=selected_chat_mode == "Classica" and current_mode == "Tutor"),
        inputs=[chat_mode, mode],
        outputs=[response_mode_tutor],
        queue=False
    )

    btn_submit.click(
        fn=process_message,
        inputs=[domanda_input, chatbot, mode, prompt_mode, codice, response_mode_tutor, chat_mode],
        outputs=[chatbot, fonzi],
        queue=True
    ).then(
        lambda: "",
        inputs=None,
        outputs=[domanda_input]
    )

    btn_clear.click(
        fn=lambda: (
            [], # chatbot history
            "", # domanda_input
            "", # codice
            "Le fonti recuperate appariranno qui.", 
            gr.update(value="Spiegazione", visible=False), 
            gr.update(value="Tutor"), # Restore mode
            gr.update(value="Dettagliata", visible=True), 
            gr.update(value="Classica", visible=True) 
        ),
        outputs=[chatbot, domanda_input, codice, fonzi, prompt_mode, mode, response_mode_tutor, chat_mode],
        queue=False
    )

demo.launch()