import pandas as pd
from rag import vector_indices, configure_query_engine, llm_tutor, reranker, TUTOR_PROMPT
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

# --- QUERY DI ESEMPIO E RISPOSTE IDEALI (personalizza qui) ---
examples = [
    {
        "question": "Cosa è una lista e come si implementa",
        "ground_truth": "Una lista è una struttura dati che memorizza una sequenza ordinata di elementi, permettendo l'accesso, l'inserimento e la rimozione. In Java, una lista può essere implementata tramite la classe LinkedList o ArrayList della libreria standard, oppure manualmente usando una classe Node che contiene un valore e un riferimento al prossimo nodo.",
    },
    {
        "question": "Differenza tra classe interna ed esterna?",
        "ground_truth": "Una classe esterna è una classe dichiarata a livello superiore, mentre una classe interna è definita all'interno di un'altra classe. Le classi interne possono accedere ai membri della classe esterna e sono spesso usate per raggruppare logicamente classi strettamente correlate.",
    },
    {
        "question": "cosa è e come si effettua l'override in Java?",
        "ground_truth": "L'override in Java è la pratica di ridefinire un metodo ereditato da una superclasse in una sottoclasse, mantenendo la stessa firma. Si effettua dichiarando il metodo con la stessa firma nella sottoclasse e usando l'annotazione @Override.",
    },
    {
        "question": "A cosa serve la parola chiave 'final' in Java?",
        "ground_truth": "La parola chiave final in Java serve a indicare che una variabile non può essere modificata dopo l'inizializzazione, un metodo non può essere sovrascritto nelle sottoclassi e una classe non può essere estesa.",
    },
    {
        "question": "cosa sono i file e come si leggono in java?",
        "ground_truth": "Un file è una risorsa di memorizzazione persistente su disco. In Java, i file si leggono usando classi come FileReader, BufferedReader o Files.readAllLines(). Ad esempio, si può usare BufferedReader reader = new BufferedReader(new FileReader(\"nomefile.txt\")); per leggere il contenuto riga per riga.",
    },
]

# --- ESECUZIONE DEL SISTEMA RAG E RACCOLTA DATI ---
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

    # NUOVE METRICHE PER IL RECUPERO
    cos_sim_context_relevance = np.nan # Rilevanza del contesto alla domanda
    cos_sim_context_recall = np.nan # Recall del contesto rispetto alla ground truth

    if contexts:
        emb_question = embed_model.get_text_embedding(question)
        
        # Context Relevance: media delle similarità tra la domanda e ogni contesto
        relevance_scores = [cosine_similarity(emb_question, embed_model.get_text_embedding(ctx)) for ctx in contexts]
        cos_sim_context_relevance = np.mean(relevance_scores) if relevance_scores else np.nan

        # Context Recall: similarità tra i contesti CONCATENATI e la ground truth
        # Assumiamo che la ground_truth dovrebbe essere "coperta" dai contesti recuperati
        # Questo è un proxy, non una vera recall binaria come in RAGAs con annotazioni
        if concatenated_contexts: # Assicurati che i contesti non siano vuoti
            cos_sim_context_recall = cosine_similarity(emb_contexts, emb_gt)

    # Altre metriche di generazione (già presenti)
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

# --- ESPORTA CSV ---
df = pd.DataFrame(rows)
df.to_csv("rag_test.csv", index=False)
print("Creato ragas_test_with_retrieval_metrics.csv con i risultati delle query e delle metriche locali.")

# Stampa media delle metriche
print("\n=== METRICHE LOCALI (media) ===")
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