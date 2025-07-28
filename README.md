# ☕ Java AI Assistant

Sistema RAG (Retrieval Augmented Generation) per assistenza alla programmazione Java.

## 🚀 Setup Rapido

### 1. Prerequisiti
- Python 3.8+
- [Ollama](https://ollama.ai/) installato

### 2. Modelli Ollama
```bash
ollama pull llama3.1:8b
ollama pull codellama:7b
```

### 3. Installazione
```bash
# Clona il repository
git clone <repository-url>
cd RAG_Java

# Crea ambiente virtuale
python -m venv .venv

# Attiva ambiente
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

### 4. Avvio
```bash
python rag.py
```

L'applicazione sarà disponibile su `http://localhost:7860`

## 📋 Funzionalità

- **Tutor Mode**: Spiegazioni concetti Java e documentazione
- **Coding Assistant**: Analisi, debug e generazione codice
- **Modalità Chat**: Conversazione con memoria
- **Modalità Classica**: Risposta singola
- **Export PDF**: Esportazione conversazioni

## 🎯 Modalità

### Tutor
- Spiegazioni didattiche
- Esempi di codice
- Link documentazione ufficiale

### Coding Assistant
- **Spiegazione**: Analisi dettagliata codice
- **Debug**: Identificazione problemi e soluzioni
- **Crea**: Generazione codice Java

## 📁 Struttura
```
RAG_Java/
├── rag.py              # Applicazione principale
├── util.py             # Configurazioni e funzioni
├── requirements.txt    # Dipendenze Python
├── createIndex.py      # Script creazione indici
└── DATA/              # Documenti e codice Java
``` 