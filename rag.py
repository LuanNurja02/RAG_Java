import os
import time
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

# --- Configurazione e Inizializzazione Globale ---
# Importa le costanti dal file util.py
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

load_dotenv(dotenv_path='pinecone_key.env') # Assicurati che questo percorso sia corretto e accessibile

# Indici Pinecone
TUTOR_INDEX_NAME = "meta-lib"
ASSISTANT_INDEX = "java-codebase"

# Configurazione del modello di embedding
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"

# Re-ranker
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
RERANK_TOP_N = 3  # nodi da usare dopo il re-ranking

# Funzione per Configurare il Query Engine
def configure_query_engine(index_instance, llm_instance, embed_model_instance, prompt_template_instance, reranker_instance):
    retriever = VectorIndexRetriever(
        index=index_instance,
        similarity_top_k=10,
        embed_model=embed_model_instance,
        sparse_top_k=2
    )

    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.80),
        reranker_instance
    ]

    response_synthesizer = get_response_synthesizer(
        llm=llm_instance,
        streaming=True,
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
    # Global initialization of LLM, embedding, re-ranker, and Pinecone indices

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("The 'PINECONE_API_KEY' environment variable is not set.")

    pc = Pinecone(api_key=api_key)

    # Embedding
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, device=embed_device)

    # Set up the re-ranker
    reranker = FlagEmbeddingReranker(
        model=RERANK_MODEL_NAME,
        top_n=RERANK_TOP_N
    )

    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        max_tokens=OLLAMA_MAX_TOKENS,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
        context_window=OLLAMA_CONTEXT_WINDOW,
        streaming=True,
        min_length=100,
        top_p=0.9,
        repeat_penalty=1.2
    )

    # Index for Tutor mode
    pinecone_index_tutor = pc.Index(TUTOR_INDEX_NAME)
    vector_store_tutor = PineconeVectorStore(pinecone_index=pinecone_index_tutor)
    storage_context_tutor = StorageContext.from_defaults(vector_store=vector_store_tutor)
    vector_indices["Tutor"] = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_tutor,
        embed_model=embed_model,
        storage_context=storage_context_tutor
    )

    # Index for Coding Assistant mode
    pinecone_index_coding_assistant = pc.Index(ASSISTANT_INDEX)
    vector_store_coding_assistant = PineconeVectorStore(pinecone_index=pinecone_index_coding_assistant)
    storage_context_coding_assistant = StorageContext.from_defaults(vector_store=vector_store_coding_assistant)
    vector_indices["Coding Assistant"] = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_coding_assistant,
        embed_model=embed_model,
        storage_context=storage_context_coding_assistant
    )

    # Prepare default query engine for Tutor mode
    query_engines["Tutor"] = configure_query_engine(
        index_instance=vector_indices["Tutor"],
        llm_instance=llm,
        embed_model_instance=embed_model,
        prompt_template_instance=TUTOR_PROMPT,
        reranker_instance=reranker
    )

except Exception as e:
    print(f"Critical error during global initialization: {str(e)}")
    raise


def process_message(message: str, history: list, mode: str, prompt_mode: str, codice: str):
    """
    Processes the user's message, updates chat history, and generates a response.
    Yields chunks of the assistant's response for streaming, and finally the sources.
    """
    # Add the user's message to history. The bot's response is initially None.
    history.append([message, None])

    # Construct the full query for the RAG engine
    codice = codice if codice is not None else ""
    if codice.strip():
        full_query = f"{message}\n\nCODICE FORNITO:\n```java\n{codice.strip()}\n```"
    else:
        full_query = message

    if not full_query.strip():
        # If the input is empty, remove the last history entry (the empty user message)
        # and return an error message.
        if history and history[-1][0] == message and history[-1][1] is None:
            history.pop()
        yield history, "Please enter at least a question or code to analyze."
        return

    current_response_text = ""
    sources_output_text = ""

    try:
        # Determine the query engine based on the selected mode and prompt mode
        if mode == "Tutor":
            current_query_engine = query_engines["Tutor"]
            print(f"💡 Executing in TUTOR mode with query: {message[:50]}...")
        else:  # mode == "Coding Assistant"
            if prompt_mode == "Spiegazione":
                selected_prompt_template = SPIEGAZIONE_CODICE_PROMPT
            elif prompt_mode == "Debug":
                selected_prompt_template = DEBUG_CODICE_PROMPT
            elif prompt_mode == "Crea":
                selected_prompt_template = CREA_CODICE_PROMPT
            else:
                selected_prompt_template = SPIEGAZIONE_CODICE_PROMPT  # DEFAULT for Coding Assistant

            current_query_engine = configure_query_engine(
                index_instance=vector_indices["Coding Assistant"],
                llm_instance=llm,
                embed_model_instance=embed_model,
                prompt_template_instance=selected_prompt_template,
                reranker_instance=reranker
            )
            print(f"💡 Executing in CODING ASSISTANT mode ({prompt_mode}) with query: {message[:50]}...")

        # Get the streaming response from the query engine
        streaming_response = current_query_engine.query(full_query)

        # Iterate over response chunks for streaming display
        if hasattr(streaming_response, 'response_gen'):
            for text_chunk in streaming_response.response_gen:
                current_response_text += text_chunk
                # Update the bot's response in the last history entry
                history[-1][1] = current_response_text
                # Return updated history (for chat display) and empty sources (sources come later)
                yield history, ""
        else:
            # Handle non-streaming response if it occurs (even if streaming=True is set)
            current_response_text = str(getattr(streaming_response, 'response', streaming_response))
            history[-1][1] = current_response_text
            yield history, ""

        # After the full response is generated, prepare and return the sources
        if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
            sources_output_text += "### Reference Documents Used\n"
            for i, node in enumerate(streaming_response.source_nodes):
                score = getattr(node, "score", None)
                content_preview = node.get_content()
                # Truncate content for display to avoid overly long source blocks
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "...\n(Content truncated)"
                score_str = f"Score: {score:.3f}" if score is not None else "Score: N/A"
                sources_output_text += f"**[{i+1}] {score_str}**\n```\n{content_preview}\n```\n\n"

        # Return the complete history and formatted sources
        yield history, sources_output_text

    except Exception as e:
        error_message = f"An error occurred: {str(e)}\nPlease try again."
        print(f"Error processing query: {str(e)}")
        # Update the bot's response in history with the error message
        history[-1][1] = error_message
        # Return history with the error and empty sources
        yield history, ""


# Gradio Interface
with gr.Blocks(theme=themes.Soft()) as demo:
    gr.Markdown("# ☕Java AI Assistant")
    gr.Markdown(
        "Benvenuto! Sono il tuo assistente per la programmazione Java. "
        "Scegli una **modalità** qui sotto e poi scrivi la tua domanda."
        " Nella modalità **Coding Assistant**, puoi anche incollare del codice Java."
    )

    with gr.Row():
        # Left Column: Controls (Mode, Code Assistance Type, Code Input)
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
                visible=False, # Initially hidden for Tutor mode
                interactive=True
            )
            codice = gr.Textbox(
                label="Codice Java (Opzionale per Coding Assistant)",
                lines=7,
                placeholder="Incolla qui il codice Java da analizzare, debuggare o su cui vuoi basare una generazione. (Ignorato in modalità Tutor)",
                interactive=True,
                visible=False # Initially hidden for Tutor mode
            )

        # Right Column: Chat Display, User Input, and Buttons
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversazione",
                elem_id="chatbot", # ID for possible CSS customizations
                height=500 # Adjust height for better chat experience
            )
            fonti = gr.Markdown(label="Documenti di Riferimento Utilizzati", value="Le fonti recuperate appariranno qui.")

            # Row for message input and buttons
            with gr.Row():
                domanda_input = gr.Textbox(
                    label="Il tuo messaggio",
                    lines=2,
                    placeholder="Digita la tua domanda qui...",
                    interactive=True,
                    scale=8 # Gives more space to the text field
                )
                # Column to hold the buttons vertically next to the textbox
                with gr.Column(scale=2, min_width=100): # Adjust scale and min_width as needed
                    btn_submit = gr.Button("Invia", variant="primary", icon="DATA/gui_icon/coffee.png")
                    btn_clear = gr.Button("Cancella", variant="secondary", icon="DATA/gui_icon/trash.png")


    # Event Handlers
    # Update visibility of prompt_mode and codice based on selected mode
    mode.change(
        fn=lambda selected_mode: (gr.update(visible=selected_mode == "Coding Assistant"), gr.update(visible=selected_mode == "Coding Assistant")),
        inputs=mode,
        outputs=[prompt_mode, codice],
        queue=False
    )

    # Handles message submission
    btn_submit.click(
        fn=process_message,
        inputs=[domanda_input, chatbot, mode, prompt_mode, codice],
        outputs=[chatbot, fonti],
        queue=True # Important for streaming and correct operation sequence
    ).then(
        # Clear the input textbox after submission
        lambda: "",
        inputs=None,
        outputs=[domanda_input]
    )

    # Handles clearing all inputs and chat history
    btn_clear.click(
        fn=lambda: (
            [], # Clear chatbot history
            "", # Clear domanda_input
            "", # Clear codice
            "Le fonti recuperate appariranno qui.", # Restore sources text
            gr.update(value="Spiegazione", visible=False), # Restore prompt_mode and hide it
            gr.update(value="Tutor") # Restore mode
        ),
        outputs=[chatbot, domanda_input, codice, fonti, prompt_mode, mode],
        queue=False
    )

demo.launch()
