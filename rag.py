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
    TUTOR_PROMPT,
    SPIEGAZIONE_CODICE_PROMPT,
    DEBUG_CODICE_PROMPT,
    CREA_CODICE_PROMPT,
    export_to_pdf,
    tutor,
    coding,
    PROMPT_CHAT
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
import json
import datetime

load_dotenv(dotenv_path='pinecone_key.env')

TUTOR_INDEX_NAME = "documenti"
ASSISTANT_INDEX = "codebase"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
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
            prefix_messages=[],
            context_template=prompt_template_instance   #aggiungere il promt
            #sistemare il prompt di chatengine
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

llm_tutor = tutor
llm_coding = coding
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


last_response_info = {}

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
        yield history
        return

    current_response_text = ""
    global last_response_info
    last_response_info = {} 

    try:
        if mode == "Tutor":
            selected_response_mode = RESPONSE_MODE_MAP.get(response_mode_tutor, ResponseMode.COMPACT)

            if chat_mode == "Chat":
                current_query_engine = configure_query_engine(
                    index_instance=vector_indices["Tutor"],
                    llm_instance=llm_tutor,  
                    prompt_template_instance=PROMPT_CHAT,
                    reranker_instance=reranker,
                    memory=chat_memory_tutor
                )
                print(f"💡 Executing in TUTOR mode (Chat) with query: {final_query[:50]}...")
                streaming_response = current_query_engine.stream_chat(full_query)
            else:
                current_query_engine = configure_query_engine(
                    index_instance=vector_indices["Tutor"],
                    llm_instance=llm_tutor,  
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
                selected_prompt_template = SPIEGAZIONE_CODICE_PROMPT # DEFAULT Coding Assistant

            if chat_mode == "Chat":
                current_query_engine = configure_query_engine(
                    index_instance=vector_indices["Coding Assistant"],
                    llm_instance=llm_coding,  
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
                    llm_instance=llm_coding,  
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
                yield history
        else:
            current_response_text = str(getattr(streaming_response, 'response', streaming_response))
            history[response_index][1] = current_response_text
            yield history

        #ultimo feedback 
        last_response_info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": full_query,
            "response": current_response_text,
            "mode": mode,
            "prompt_mode": prompt_mode if mode == "Coding Assistant" else None,
            "response_mode_tutor": response_mode_tutor if mode == "Tutor" else None,
            "chat_mode": chat_mode,
            "feedback_rating": None,
            "source_nodes": []
        }

        if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
            # Stampa i documenti recuperati nel terminale come debug, per verede se sono realmente inerenti alla query
            print("\n" + "="*80)
            print("DOCUMENTI DI RIFERIMENTO UTILIZZATI")
            print("="*80)
            for i, node in enumerate(streaming_response.source_nodes):
                score = getattr(node, "score", None)
                content_preview = node.get_content()
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "...\n(Contenuto troncato)"
                score_str = f"Score: {score:.3f}" if score is not None else "Score: N/A"
                print(f"\n DOCUMENTO [{i+1}] {score_str}")
                print("-" * 60)
                print(content_preview)
                print("-" * 60)
                
                last_response_info["source_nodes"].append({
                    "content_preview": content_preview,
                    "score": score
                })
            print("="*80 + "\n")

        yield history

    except Exception as e:
        error_message = f"Si è verificato un errore: {str(e)}\nPer favore riprova."
        print(f"Error processing query: {str(e)}")
        history[-1][1] = error_message
        yield history

def save_feedback(rating: int):
    global last_response_info
    if not last_response_info:
        return "Nessuna risposta precedente da valutare."

    last_response_info["feedback_rating"] = rating
    
    #salvo i feedback
    feedback_dir = "feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    feedback_filename = os.path.join(feedback_dir, "feedback.json")
    
    feedback_data = []
    if os.path.exists(feedback_filename):
        with open(feedback_filename, 'r', encoding='utf-8') as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                feedback_data = [] 

    feedback_data.append(last_response_info)

    with open(feedback_filename, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, indent=4, ensure_ascii=False)
    
    # Reset dopo il salvataggio
    last_response_info = {}
    return f"Grazie per il tuo feedback! Voto: {rating} stella/e salvato."


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
                visible=True, 
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
                label="Codice",
                lines=7,
                placeholder=""" public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
""",
                interactive=True,
                visible=False,
                autoscroll=True,
                autofocus=True,
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversazione",
                elem_id="chatbot",
                height=500
                
            )

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
                    btn_export = gr.Button("📄 Esporta PDF", variant="secondary")
            
            # Feedback 
            gr.Markdown("### Valuta l'ultima risposta")
            with gr.Row():
                feedback_rating = gr.Radio(
                    choices=[1, 2, 3, 4, 5],
                    label="Voto (1-5 Stelle)",
                    type="value",
                    visible=False, 
                    interactive=True
                )
                btn_feedback = gr.Button("Invia Feedback", visible=False, variant="secondary")
            feedback_status = gr.Markdown(value="", visible=False)


    #salvataggio della conversazione in formato pdf
    pdf_download = gr.File(label="Scarica PDF", visible=False)
    export_status = gr.Markdown(value="", visible=False)

    mode.change(
        fn=lambda selected_mode: (
            gr.update(visible=selected_mode == "Coding Assistant"), 
            gr.update(visible=selected_mode == "Coding Assistant"), 
            gr.update(visible=selected_mode == "Tutor") 
        ),
        inputs=mode,
        outputs=[prompt_mode, codice, response_mode_tutor],
        queue=False
    )

    chat_mode.change(
        fn=lambda selected_chat_mode, current_mode: gr.update(visible=selected_chat_mode == "Classica" and current_mode == "Tutor"),
        inputs=[chat_mode, mode],
        outputs=[response_mode_tutor],
        queue=False
    )

    btn_submit.click(
        fn=process_message,
        inputs=[domanda_input, chatbot, mode, prompt_mode, codice, response_mode_tutor, chat_mode],
        outputs=[chatbot],
        queue=True
    ).then(
        lambda: (
            "", # Clear input
            gr.update(visible=True), 
            gr.update(visible=True) 
        ),
        inputs=None,
        outputs=[domanda_input, feedback_rating, btn_feedback]
    )

    btn_clear.click(
        fn=lambda: (
            [], # chatbot history
            "", # domanda_input
            "", # codice
            gr.update(value="Spiegazione", visible=False),
            gr.update(value="Tutor"), 
            gr.update(value="Dettagliata", visible=True),
            gr.update(value="Classica", visible=True),
            gr.update(visible=False), 
            gr.update(value="", visible=False), 
            gr.update(value=None, visible=False), 
            gr.update(visible=False), 
            gr.update(value="", visible=False) 
        ),
        outputs=[
            chatbot, domanda_input, codice, prompt_mode, mode,
            response_mode_tutor, chat_mode, pdf_download, export_status,
            feedback_rating, btn_feedback, feedback_status
        ],
        queue=False
    )

    # Export PDF functionality
    def handle_export(history):
        filename, message = export_to_pdf(history)
        if filename:
            return (
                gr.update(value=filename, visible=True),
                gr.update(value=f" {message}", visible=True)
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(value=f" {message}", visible=True)
            )

    btn_export.click(
        fn=handle_export,
        inputs=[chatbot],
        outputs=[pdf_download, export_status],
        queue=False
    )
    
    # Feedback button click
    btn_feedback.click(
        fn=save_feedback,
        inputs=[feedback_rating],
        outputs=[feedback_status]
    ).then(
        lambda: (
            gr.update(value=None, visible=False), 
            gr.update(visible=False) 
        ), 
        inputs=None,
        outputs=[feedback_rating, btn_feedback]
    )

if __name__ == "__main__":
    demo.launch()