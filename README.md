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
  Copy code
  python -m venv env
  source env/bin/activate  # On Windows, use `env\Scripts\activate`
  pip install -r requirements.txt
  ```

3. 🚀 Usage
Data Preprocessing
Use the datasets.py script to handle loading and preparing data for training. Ensure that your datasets are correctly formatted before proceeding:
 ```bash
  python src/datasets.py
 ```
 
4. Training
All training processes are conducted using Jupyter notebooks located in the notebooks/ directory. You can run the notebooks to train the model from scratch or fine-tune it on specific downstream tasks. 

5. Model Implementation
The implementation of the CamemBERT-like model from scratch is located in the src/model/ directory. You can directly use this model in your experiments:

```bash
  from src.model.camembert_model import CamemBERTBase
```

6. Evaluation
Evaluate the performance of your trained model on specific tasks using the evaluation notebooks or scripts:

notebooks/evaluate_nli.ipynb
notebooks/evaluate_ner.ipynb

7. 📊 Results
The project achieves competitive performance on the following tasks:

NLI (Natural Language Inference): Accuracy of 85% on the XNLI dataset.
NER (Named Entity Recognition): F1-score of 91% on the CoNLL-2003 dataset.
POS Tagging and Dependency Parsing: Results comparable to the original CamemBERT paper across multiple French treebanks (e.g., GSD, Sequoia).

9. 🤝 Contributors

- Noureddine Khaous
- Raid Salhi
- Amine Ouguouenoune
- Ramy Larabi