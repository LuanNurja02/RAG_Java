import pandas as pd
from rag import vector_indices, configure_query_engine, llm_tutor, reranker
from util import TUTOR_PROMPT
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from bert_score import score as bert_score
import sacrebleu


nltk.download('punkt', quiet=True)

# Inizializza modello di embedding come in rag.py
embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-small-v2")

# Funzione per similarità coseno
from numpy.linalg import norm
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

#dataset per domande e risposte usate per valutare il sistema rag---
examples = [
    {
        "question": "Puoi spiegarmi cos'è il polimorfismo in Java?",
        "ground_truth": "Il polimorfismo in Java è un concetto della programmazione orientata agli oggetti che consente a oggetti di classi diverse di essere trattati come oggetti della stessa superclasse. Ciò è reso possibile attraverso l'ereditarietà e l'uso delle interfacce. Esistono due tipi principali di polimorfismo: statico (overloading) e dinamico (overriding)."
    },
    {
        "question": "Che ruolo ha l'ereditarietà nella programmazione Java?",
        "ground_truth": "L'ereditarietà in Java permette a una classe di acquisire le proprietà e i metodi di un'altra classe. La classe che eredita prende il nome di sottoclasse, mentre quella da cui eredita è detta superclasse. È uno dei pilastri della programmazione a oggetti, facilitando il riuso del codice e la creazione di gerarchie di classi."
    },
    {
        "question": "Perché si usa il blocco try-catch in Java?",
        "ground_truth": "Il costrutto try-catch in Java serve per gestire le eccezioni, ovvero situazioni anomale che possono verificarsi durante l'esecuzione del programma. Il blocco try contiene il codice che può generare un'eccezione, mentre il blocco catch intercetta e gestisce l'eccezione, evitando che il programma termini in modo anomalo."
    },
    {
        "question": "Qual è la differenza tra interfaccia e classe astratta?",
        "ground_truth": "Una interfaccia in Java definisce un contratto che una classe deve rispettare, specificando solo la firma dei metodi. Una classe astratta può invece contenere sia metodi astratti che implementati. Le interfacce supportano l'ereditarietà multipla, mentre le classi astratte no. Dal Java 8, le interfacce possono anche avere metodi di default."
    },
    {
        "question": "Come si dichiarano e usano classi interne in Java?",
        "ground_truth": "Una classe interna è una classe dichiarata all'interno di un'altra classe. Può accedere ai membri (anche privati) della classe esterna. Si usa per raggruppare logicamente classi correlate. Le classi esterne, invece, sono dichiarate a livello di file. Le classi interne possono essere statiche o non statiche."
    },
    {
        "question": "In che modo si può scrivere un'espressione lambda in Java?",
        "ground_truth": "Le espressioni lambda in Java sono introdotte a partire da Java 8 e permettono di scrivere funzioni anonime. La sintassi base è (parametri) -> { corpo }. Sono spesso usate con le interfacce funzionali, come Runnable o Comparator, per rendere il codice più conciso e leggibile."
    },
    {
        "question": "Per cosa viene usato un costruttore in Java?",
        "ground_truth": "Il costruttore è un metodo speciale usato per inizializzare un oggetto quando viene creato. Ha lo stesso nome della classe e non ha un tipo di ritorno. Può essere sovraccaricato per creare oggetti con diverse inizializzazioni."
    },
    {
        "question": "Come si fa l'override di un metodo in Java?",
        "ground_truth": "L'override (sovrascrittura) di un metodo in Java si effettua quando una sottoclasse fornisce una nuova implementazione di un metodo ereditato dalla superclasse. Si usa l'annotazione @Override per indicarlo chiaramente. Il metodo deve avere la stessa firma e non può essere più restrittivo nei modificatori di accesso."
    },
    {
        "question": "Qual è il modo corretto per leggere file binari in Java?",
        "ground_truth": "Per leggere file binari in Java si usano classi come FileInputStream o DataInputStream. Ad esempio, con FileInputStream fis = new FileInputStream(\"file.dat\"); si può leggere byte per byte. Questi flussi permettono di leggere dati grezzi in formato binario."
    },
    {
        "question": "Come funziona il metodo Math.sin() in Java?",
        "ground_truth": "La classe Math in Java fornisce metodi statici per operazioni matematiche comuni. Il metodo Math.sin(double a) restituisce il seno dell'angolo specificato in radianti. Per esempio, Math.sin(Math.PI/2) restituisce 1.0."
    }
]



index = vector_indices["Tutor"]
query_engine = configure_query_engine(
    index_instance=index,
    llm_instance=llm_tutor,
    prompt_template_instance=TUTOR_PROMPT,
    reranker_instance=reranker
)

rows = []
for ex in examples:
    question = ex["question"]
    ground_truth = ex["ground_truth"]
    
    # Retrieval + generazione esattamente come in rag.py
    response = query_engine.query(question)
    
    if hasattr(response, "source_nodes") and response.source_nodes:
        contexts = [node.get_content()[:200] for node in response.source_nodes]
    else:
        contexts = []
        
    if hasattr(response, "response"):
        answer = str(response.response)
    else:
        answer = str(response)

    # Calcolo metriche per la Generazione (Answer vs Ground Truth)
    emb_gt = embed_model.get_text_embedding(ground_truth)
    emb_ans = embed_model.get_text_embedding(answer)
    cos_sim_gt_ans = cosine_similarity(emb_gt, emb_ans)

    # Calcolo metrica per la Groundedness (Answer vs Contexts)
    cos_sim_context_ans = np.nan 
    concatenated_contexts = ""
    if contexts:
        concatenated_contexts = " ".join(contexts)
        emb_contexts = embed_model.get_text_embedding(concatenated_contexts)
        cos_sim_context_ans = cosine_similarity(emb_contexts, emb_ans)

    cos_sim_context_relevance = np.nan # Rilevanza del contesto alla domanda
    cos_sim_context_recall = np.nan # Recall del contesto rispetto alla ground truth

    if contexts:
        emb_question = embed_model.get_text_embedding(question)
        
        # Context Relevance: media delle similarità tra la domanda e ogni contesto
        relevance_scores = [cosine_similarity(emb_question, embed_model.get_text_embedding(ctx)) for ctx in contexts]
        cos_sim_context_relevance = np.mean(relevance_scores) if relevance_scores else np.nan

        # Context Recall: similarità tra i contesti CONCATENATI e la ground truth
        if concatenated_contexts: 
            cos_sim_context_recall = cosine_similarity(emb_contexts, emb_gt)


    ref = [nltk.word_tokenize(ground_truth.lower())]
    hyp = nltk.word_tokenize(answer.lower())
    bleu = sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
    
    lev_dist = Levenshtein.distance(ground_truth, answer)
    lev_norm = 1 - lev_dist / max(len(ground_truth), len(answer), 1)
    
    P, R, F1 = bert_score([answer], [ground_truth], lang="it", rescale_with_baseline=False)
    bert_p = float(P[0])
    bert_r = float(R[0])
    bert_f1 = float(F1[0])
    
    ter = sacrebleu.metrics.TER().corpus_score([answer], [[ground_truth]]).score / 100.0

    rows.append({
        "question": question,
        "ground_truth": ground_truth,
        "contexts": contexts,
        "answer": answer,
        "cosine_similarity_gt_ans": cos_sim_gt_ans,
        "cosine_similarity_context_ans": cos_sim_context_ans,
        "cosine_similarity_context_relevance": cos_sim_context_relevance, # NUOVA
        "cosine_similarity_context_recall": cos_sim_context_recall,       # NUOVA
        "bleu": bleu,
        "levenshtein_norm": lev_norm,
        "bertscore_p": bert_p,
        "bertscore_r": bert_r,
        "bertscore_f1": bert_f1,
        "ter": ter,
    })

#saltavaggio
df = pd.DataFrame(rows)
df.to_csv("rag_test.csv", index=False)
print("Creato ragas_test_with_retrieval_metrics.csv con i risultati delle query e delle metriche locali.")

# Stampa media delle metriche
print("\n=== METRICHE ===")
print("Cosine similarity (Ground Truth vs Answer):", df["cosine_similarity_gt_ans"].mean())
print("Cosine similarity (Context vs Answer):", df["cosine_similarity_context_ans"].mean())
print("Cosine similarity (Context Relevance - Question vs Contexts):", df["cosine_similarity_context_relevance"].mean())
print("Cosine similarity (Context Recall - Contexts vs Ground Truth):", df["cosine_similarity_context_recall"].mean())
print("BLEU:", df["bleu"].mean())
print("Levenshtein (normalizzato):", df["levenshtein_norm"].mean())
print("BERTScore P:", df["bertscore_p"].mean())
print("BERTScore R:", df["bertscore_r"].mean())
print("BERTScore F1:", df["bertscore_f1"].mean())
print("TER:", df["ter"].mean())