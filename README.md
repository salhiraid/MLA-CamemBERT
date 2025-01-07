# MLA-CamemBERT

**MLA-CamemBERT** is a project aimed at reproducing and adapting the **CamemBERT** model, a French-optimized version of RoBERTa.

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


🚀 Usage
1. Data Preprocessing
The script datasets.py handles loading and preparing data for training. Make sure your datasets are properly formatted.

2. Training
Run the training process using the notebooks in the notebooks/ directory or via the source scripts in src/:

    ```bash
    Copy code
    python src/train.py