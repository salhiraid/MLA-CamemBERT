# MLA-CamemBERT

**MLA-CamemBERT** is a project aimed at reproducing and adapting the **CamemBERT** model, a French-optimized version of RoBERTa. This project explores techniques to avoid expensive pre-training while maintaining high performance using approaches like **FOCUS** and fine-tuning on specific tasks.

---

## 📚 **Description**

This project provides a complete pipeline to:
- Load and preprocess French-language datasets.
- Adapt a multilingual model to the French language.
- Fine-tune the model on various NLP tasks, including:
  - **POS Tagging (Part-of-Speech Tagging)**
  - **Dependency Parsing**
  - **Natural Language Inference (NLI)**
  - **Named Entity Recognition (NER)**

The model is designed to be efficient to train and easily integratable into NLP applications.

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/salhiraid/MLA-CamemBERT.git
   cd MLA-CamemBERT

2. Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
