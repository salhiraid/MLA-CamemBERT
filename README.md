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

  🚀 Usage
Data Preprocessing
Use the datasets.py script to handle loading and preparing data for training. Ensure that your datasets are correctly formatted before proceeding:

  ```bash
  Copy code
  python src/datasets.py

3. Training
All training processes are conducted using Jupyter notebooks located in the notebooks/ directory. You can run the notebooks to train the model from scratch or fine-tune it on specific downstream tasks. For example:

notebooks/train_mlm.ipynb for pretraining on Masked Language Modeling.
notebooks/finetune_nli.ipynb for fine-tuning on Natural Language Inference.
notebooks/finetune_ner.ipynb for Named Entity Recognition.
Model Implementation
The implementation of the CamemBERT-like model from scratch is located in the src/model/ directory. You can directly use this model in your experiments:

python
Copy code
from src.model.camembert_model import CamemBERTBase
Evaluation
Evaluate the performance of your trained model on specific tasks using the evaluation notebooks or scripts:

notebooks/evaluate_nli.ipynb
notebooks/evaluate_ner.ipynb
📊 Results
The project achieves competitive performance on the following tasks:

NLI (Natural Language Inference): Accuracy of 85% on the XNLI dataset.
NER (Named Entity Recognition): F1-score of 91% on the CoNLL-2003 dataset.
POS Tagging and Dependency Parsing: Results comparable to the original CamemBERT paper across multiple French treebanks (e.g., GSD, Sequoia).
🧪 Experimentation
Datasets

OSCAR Dataset: Used for pretraining the model on French text.
UD Treebanks: Used for POS tagging and dependency parsing.
XNLI Dataset: Used for the NLI task.
CoNLL-2003 Dataset: Used for NER.
Pretraining
The model was pretrained on a 4GB subset of the French OSCAR dataset using the MLM (Masked Language Modeling) objective.

Fine-Tuning
Fine-tuning was performed on each downstream task using task-specific datasets and metrics (e.g., F1-score for NER, Accuracy for NLI).

📂 Project Structure
bash
Copy code
MLA-CamemBERT/
├── data/                 # Contains raw and processed datasets
├── notebooks/            # Jupyter notebooks for training and fine-tuning
│   ├── train_mlm.ipynb   # Notebook for pretraining on MLM
│   ├── finetune_nli.ipynb # Notebook for fine-tuning on NLI
│   ├── finetune_ner.ipynb # Notebook for fine-tuning on NER
│   ├── evaluate_nli.ipynb # Notebook for evaluating NLI task
├── src/                  # Source code for model and utilities
│   ├── model/            # Implementation of the CamemBERT model
│   ├── datasets.py       # Data preprocessing and loading
│   ├── evaluate.py       # Evaluation script
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
🤝 Contributors
Noureddine Khaous
Raid Salhi
Amine Ouguouenoune
Ramy Larabi