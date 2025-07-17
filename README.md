# ComplexNER + Chatbot

**ComplexNER** is a repository dedicated to Named Entity Recognition (NER) using BERT (Bidirectional Encoder Representations from Transformers) combined with Conditional Random Fields (CRF). The project provides implementations for NER in both English and Hindi languages and explores advanced methods to tackle complex NER scenarios like nested and overlapping entities.

---

## Project Overview

Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that identifies and categorizes entities such as people, organizations, and locations in text. This repository focuses on improving NER accuracy for complex scenarios including:
- **Nested Entities**: Entities that contain other entities (e.g., "Bank of New York Mellon").
- **Overlapping Entities**: Terms belonging to multiple categories.
- **Multilingual Contexts**: Tackling language diversity and low-resource settings.

Our approach combines **transformer-based models** (e.g., BERT, XLM-RoBERTa) with **CRF layers**, enhancing the system's ability to capture sequence dependencies and improve accuracy across diverse datasets like **MultiCoNER V2**, **CoNLL-2003**, and **WikiGold**.

---

## 🔍 New: Generative AI-Powered Document Chatbot (RAG + NER)

We now support a **document-based Q&A chatbot** powered by:

- 📄 **Document ingestion** (PDF, TXT)
- 🔍 **FAISS vector store** for efficient chunk retrieval
- 🤖 **GPT-4 or GPT-3.5** for natural language answers
- 🧠 **BERT+CRF NER** applied on context chunks for entity-aware reasoning

This allows users to:
- Ask natural questions about documents.
- Get answers grounded in retrieved text.
- Preserve entity structure for domain-specific applications.

---

## 💡 Architecture

```text
[ Document Upload ]
        ↓
[ Text Chunking ]
        ↓
[ Sentence Embedding ]
        ↓
[ FAISS Vector Store ]
        ↓
[ Query → Embedding → Retrieve Top-k Chunks ]
        ↓
[ BERT+CRF NER Inference on Chunks ]
        ↓
[ Prompt Built with Entity-Tags + Context ]
        ↓
[ GPT-4 Answer Generation ]
```

---

## 🧪 How to Use the Chatbot

```bash
# Step 1: Install deps
pip install -r chatbot/requirements.txt

# Step 2: Set your OpenAI key
export OPENAI_API_KEY="sk-..."

# Step 3: Run chatbot
python chatbot/document_loader.py       # Load & chunk text
python chatbot/embedder.py              # Embed & build FAISS index
python chatbot/rag_chat.py              # Ask: GPT with NER-powered retrieval
```

---

## 📁 Chatbot Code Structure

```
chatbot/
├── document_loader.py   # PDF/TXT ingestion & chunking
├── embedder.py          # SentenceTransformers + FAISS
├── retriever.py         # Query → Top-k chunk retrieval
├── ner_infer.py         # BERT-CRF model inference
├── rag_chat.py          # Full pipeline with GPT answer generation
├── requirements.txt
```

---

## Key Features

- **Advanced Architecture**: Combines BERT embeddings with CRF for structured predictions.
- **Dataset Support**:
  - WikiGold: High-quality annotations for general-purpose NER.
  - CoNLL-2003: Standard benchmark for NER research.
  - MultiCoNER V2: Tackles complex NER tasks with multilingual and fine-grained entity labels.
- **Multilingual Capabilities**: Includes support for English and Hindi datasets.
- **Model Checkpointing**: Implements robust saving mechanisms for reproducibility.

---

## Methods

### 1. Baseline Models and Enhancements
- Initial experiments with **BERT (Base, Cased)** on CoNLL-2003 established a strong contextual baseline.
- Integration of **Conditional Random Fields (CRF)** to model dependencies between entity tags, achieving improved performance.
- **Scaling to BERT-Large-Cased** provided additional capacity for capturing intricate patterns.

### 2. Transition to Advanced Models
- **XLM-RoBERTa** was evaluated for multilingual scenarios, especially Hindi datasets, demonstrating reasonable performance but lagging behind BERT-Large-Cased with CRF for English.

### 3. Focus on MultiCoNER V2
- Employed the English subset of MultiCoNER V2 for final model training and evaluation.
- Tackled complex challenges like ambiguous contexts, overlapping entities, and low-resource scenarios.

---

## Datasets

1. **WikiGold**: 
   - Derived from Wikipedia articles.
   - Annotations for general-purpose entities like Person (PER), Location (LOC), and Organization (ORG).

2. **CoNLL-2003**: 
   - Newswire dataset featuring BIO tagging.
   - Four main categories: Person (PER), Location (LOC), Organization (ORG), Miscellaneous (MISC).

3. **MultiCoNER V2**:
   - Multilingual dataset with fine-grained entities across 14 languages.
   - Rich taxonomy includes Medical, Creative Works, Product, and more.

### Preprocessing
- **Tokenization**: Subword tokenization using the BERT tokenizer.
- **Label Alignment**: Ensures compatibility between input tokens and entity labels.
- **Data Splits**: Stratified sampling into training, validation, and test sets.

---

## Results

| **Dataset**          | **Language** | **F1 Score** | **Precision** | **Recall** |
|-----------------------|--------------|--------------|---------------|------------|
| CoNLL-2003 (Baseline) | English      | 91.0%        | 90.5%         | 91.5%      |
| CoNLL-2003 (CRF)      | English      | 94.0%        | 93.7%         | 94.3%      |
| WikiGold              | English      | 94.0%        | 93.9%         | 94.2%      |
| MultiCoNER V2         | English      | 66.0% (Micro) | 65.8%         | 66.3%      |

---

## Future Work

- **UI Integration**: Streamlit or Gradio interface for chatbot.
- **NER Heatmap Visualization**: Visualize tagged entity spans in retrieved context.
- **Open-Source LLM Support**: Add support for LLaMA, Mistral, Claude, etc.
- **Confidence Filtering**: Add thresholds for entity tags.

---

## Conclusion

This project presents a robust framework for tackling complex NER tasks while extending its utility through a retrieval-augmented chatbot. The integration of BERT-CRF with GPT-based generation allows entity-aware reasoning over unstructured documents, opening the door for enterprise applications in legal, finance, medical, and multilingual domains.

---
