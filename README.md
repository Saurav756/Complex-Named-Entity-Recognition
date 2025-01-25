# ComplexNER

**ComplexNER** is a repository dedicated to Named Entity Recognition (NER) using BERT (Bidirectional Encoder Representations from Transformers) combined with Conditional Random Fields (CRF). The project provides implementations for NER in both English and Hindi languages and explores advanced methods to tackle complex NER scenarios like nested and overlapping entities.

---

## Project Overview

Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that identifies and categorizes entities such as people, organizations, and locations in text. This repository focuses on improving NER accuracy for complex scenarios including:
- **Nested Entities**: Entities that contain other entities (e.g., "Bank of New York Mellon").
- **Overlapping Entities**: Terms belonging to multiple categories.
- **Multilingual Contexts**: Tackling language diversity and low-resource settings.

Our approach combines **transformer-based models** (e.g., BERT, XLM-RoBERTa) with **CRF layers**, enhancing the system's ability to capture sequence dependencies and improve accuracy across diverse datasets like **MultiCoNER V2**, **CoNLL-2003**, and **WikiGold**.

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

- **Optimization**: Explore methods like knowledge distillation and quantization to make the framework suitable for resource-constrained environments.
- **Uncertainty Quantification**: Integrate techniques to handle ambiguous cases and provide confidence scores for predictions.
- **Nested Entity Recognition**: Improve handling of nested and overlapping entities with advanced architectures.

---

## Conclusion

This project presents a robust framework for tackling complex NER tasks, demonstrating significant improvements over baseline methods. By leveraging transformer-based embeddings and CRF for structured prediction, the approach addresses challenges in multilingual and domain-specific contexts, paving the way for future advancements in NER systems.

---
