from llama_index.core.prompts import PromptTemplate
import tempfile
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY

# Parametri del modello LLM (Ollama)
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_TOKENS = 14000
OLLAMA_CONTEXT_WINDOW = 14000
OLLAMA_REQUEST_TIMEOUT = 600

# --- Prompt per le Diverse Modalità ---

TUTOR_PROMPT = PromptTemplate(
    """Sei un tutor esperto di programmazione Java. Devi fornire una spiegazione dettagliata e didattica basata ESCLUSIVAMENTE sulle informazioni fornite nel contesto.
    Parla quindi delle informazioni recuperate e rispondi alla domanda, facendo esempi di codice java. In aggiunta fornisci il link della documentazione ufficiale,dove approfondire la risposta.
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



def export_to_pdf(history):
    """
    Esporta la conversazione in PDF
    """
    if not history or len(history) == 0:
        return None, "Nessuna conversazione da esportare."
    
    try:
        # Crea un file temporaneo
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_filename = temp_file.name
        temp_file.close()
        
        # Configura il documento PDF
        doc = SimpleDocTemplate(
            temp_filename,
            pagesize=A4,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Stili
        styles = getSampleStyleSheet()
        
        # Stile per il titolo
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=HexColor('#2E4057'),
            spaceAfter=20,
            alignment=TA_LEFT
        )
        
        # Stile per la data
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=HexColor('#666666'),
            spaceAfter=20,
            alignment=TA_LEFT
        )
        
        # Stile per le domande
        question_style = ParagraphStyle(
            'QuestionStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=HexColor('#1E3A8A'),
            fontName='Helvetica-Bold',
            spaceAfter=10,
            spaceBefore=15,
            leftIndent=0,
            alignment=TA_LEFT
        )
        
        # Stile per le risposte (testo normale)
        answer_style = ParagraphStyle(
            'AnswerStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=black,
            spaceAfter=15,
            alignment=TA_JUSTIFY,
            leftIndent=10
        )
        
        # Stile per il codice
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Code'],
            fontSize=9,
            fontName='Courier',
            textColor=HexColor('#03229c'),
            backColor=HexColor('#F3F4F6'),
            borderColor=HexColor('#E5E7EB'),
            borderWidth=1,
            borderPadding=8,
            spaceAfter=10,
            spaceBefore=5,
            leftIndent=20,
            rightIndent=10
        )
        
        # Contenuto del PDF
        story = []
        
        # Titolo
        story.append(Paragraph("☕ Java AI Assistant - Conversazione", title_style))
        
        # Data di esportazione
        current_date = datetime.now().strftime("%d/%m/%Y alle %H:%M")
        story.append(Paragraph(f"Esportato il: {current_date}", date_style))
        
        # Elabora ogni scambio della conversazione
        for i, (question, answer) in enumerate(history, 1):
            if question and answer:
                # Numero e domanda
                story.append(Paragraph(f"<b>Domanda {i}:</b>", question_style))
                story.append(Paragraph(question, answer_style))
                
                # Risposta
                story.append(Paragraph(f"<b>Risposta:</b>", question_style))
                
                # Processa la risposta per separare testo e codice
                process_answer_for_pdf(answer, answer_style, code_style, story)
                
                # Separatore tra conversazioni
                if i < len(history):
                    story.append(Spacer(1, 20))
        
        # Genera il PDF
        doc.build(story)
        
        return temp_filename, "PDF esportato con successo!"
        
    except Exception as e:
        return None, f"Errore durante l'esportazione: {str(e)}"

def process_answer_for_pdf(answer_text, text_style, code_style, story):
    """
    Processa il testo della risposta per distinguere tra testo normale e blocchi di codice
    """
    if not answer_text:
        return
    
    # Pattern per identificare blocchi di codice Java (```java ... ``` o ``` ... ```)
    code_pattern = r'```(?:java)?\s*\n?(.*?)\n?```'
    
    # Dividi il testo in parti
    parts = re.split(code_pattern, answer_text, flags=re.DOTALL)
    
    for i, part in enumerate(parts):
        if not part.strip():
            continue
            
        # I blocchi di codice sono negli indici dispari dopo lo split
        if i % 2 == 1:  # Blocco di codice
            # Pulisci il codice
            clean_code = part.strip()
            if clean_code:
                story.append(Preformatted(clean_code, code_style))
        else:  # Testo normale
            # Dividi in paragrafi e processa
            paragraphs = part.split('\n\n')
            for paragraph in paragraphs:
                clean_paragraph = paragraph.strip()
                if clean_paragraph:
                    # Gestisci evidenziazioni e formattazioni markdown di base
                    formatted_paragraph = format_text_for_pdf(clean_paragraph)
                    if formatted_paragraph:
                        story.append(Paragraph(formatted_paragraph, text_style))

def format_text_for_pdf(text):
    """
    Converte alcune formattazioni markdown di base in HTML per ReportLab
    """
    if not text:
        return ""
    
    # Sostituisci **testo** con <b>testo</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # Sostituisci *testo* con <i>testo</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    # Sostituisci `codice` con formattazione monospace
    text = re.sub(r'`([^`]+)`', r'<font name="Courier" color="#03229c">\1</font>', text)
    
    # Gestisci liste semplici (- item)
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            formatted_lines.append(f"• {line[2:]}")
        elif line.startswith('* '):
            formatted_lines.append(f"• {line[2:]}")
        else:
            formatted_lines.append(line)
    
    return '<br/>'.join(formatted_lines)