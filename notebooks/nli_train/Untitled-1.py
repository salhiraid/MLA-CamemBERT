# %%
import torch
from torch import nn
from transformers import CamembertModel, CamembertTokenizer, CamembertConfig

from torch import nn, Tensor
from torch.nn.functional import softmax

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# %% [markdown]
# ## 1. Prepare Data :

# %%
data_path = "../../data/xnli"
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# %%
class XNLIDataset(Dataset):
    def __init__(self, split="train", language="fr", tokenizer=tokenizer, cache_directory="../data/xnli", max_length=128):
        """
        Dataset PyTorch pour le dataset XNLI.

        Args:
            split (str): Partition des données ("train", "test", "validation").
            language (str): Langue cible.
            cache_directory (str): Répertoire pour stocker le dataset téléchargé.
            max_length (int): Longueur maximale pour le padding/truncation.
        """
        super(XNLIDataset, self).__init__()
        self.split = split
        self.language = language
        self.cache_directory = cache_directory
        self.max_length = max_length

        # Charger les données et le tokenizer
        self.data = load_dataset(
            "facebook/xnli",
            name=self.language,
            cache_dir=self.cache_directory
        )[self.split]  # Charger uniquement la partition demandée

        self.tokenizer = tokenizer #CamembertTokenizer.from_pretrained("camembert-base")

    def __len__(self):
        """Retourne la taille du dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Récupère un échantillon spécifique.

        Args:
            idx (int): Index de l'échantillon.

        Returns:
            dict: Contient les `input_ids`, `attention_mask` et `label`.
        """
        example = self.data[idx]
        inputs = self.tokenizer(
            example["premise"],
            example["hypothesis"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Ajouter les labels
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Enlever la dimension batch
        inputs["label"] = torch.tensor(example["label"], dtype=torch.long)

        return inputs

# %%
xnli_train_dataset = XNLIDataset(split="train", language="fr", cache_directory=data_path, max_length=32)
xnli_val_dataset = XNLIDataset(split="validation", language="fr", cache_directory=data_path, max_length=32)

train_loader = DataLoader(xnli_train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(xnli_val_dataset, batch_size=2, shuffle=False)

# %% [markdown]
# ## 2. Prepare the model

# %%
class CamemBERTBaseModel(nn.Module):
    def __init__(self, model_path: str, trainable: bool = False):
        """
        Initialize the base CamemBERT model.
        :param model_path: Path to the pre-trained CamemBERT model.
        """
        super(CamemBERTBaseModel, self).__init__()
        self.base_model = CamembertModel.from_pretrained(model_path)
        self.tranaible = trainable
        self.config = CamembertConfig()
        #self.config = CamembertModel.from_pretrained(model_path).config

        if not trainable:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()
        else :
            self.base_model.train()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the base model.
        :param input_ids: Tensor of token IDs.
        :param attention_mask: Tensor of attention masks.
        :return: Last hidden states from the base model.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def get_hidden_size(self) -> int:
        """
        Get the hidden size of the base model for dynamically attaching heads.
        :return: Hidden size of the CamemBERT model.
        """
        return self.config.hidden_size

# %%
class NLIHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 3):
        """
        Initialize the NLI head.
        :param hidden_size: Hidden size of the base model's output (e.g., 768 for CamemBERT).
        :param num_labels: Number of labels for NLI (default: 3 - coherent, neutral, contradictory).
        """
        super(NLIHead, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, cls_output: Tensor) -> Tensor:
        """
        Forward pass for the NLI head.
        :param cls_output: Tensor containing the [CLS] token representation (batch_size, hidden_size).
        :return: Logits for each class (batch_size, num_labels).
        """
        return self.classifier(cls_output)

# %%
class NLIFinetuningModel(nn.Module):
    def __init__(self, base_model: CamemBERTBaseModel, num_labels: int = 3):
        """
        Initialize the NLI fine-tuning model.
        :param base_model: Instance of the base CamemBERT model.
        :param num_labels: Number of labels for NLI.
        """
        super(NLIFinetuningModel, self).__init__()
        self.base_model = base_model 
        self.nli_head = NLIHead(base_model.get_hidden_size(), num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass for NLI fine-tuning.
        :param input_ids: Tensor of token IDs.
        :param attention_mask: Tensor of attention masks.
        :param labels: Optional tensor of labels (batch_size).
        :return: Dictionary containing logits and optionally loss.
        """
        # Get last hidden states from the base model
        hidden_states = self.base_model(input_ids=input_ids, attention_mask=attention_mask) # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)

        # Extract the [CLS] token's representation
        cls_output = hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Pass through the NLI head
        logits = self.nli_head(cls_output)  # Shape: (batch_size, num_labels)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}

# %% [markdown]
# ## 3. Train 

# %%
model_path = "../../../models/4gb_oscar"
# 1. Load the base model
base_camembert = CamembertModel.from_pretrained(model_path)
# 2. Load the NLI model :
nli_camembert = NLIFinetuningModel(base_model=CamemBERTBaseModel(model_path), num_labels=3)

optimizer = torch.optim.Adam(nli_camembert.parameters(), lr=5e-5)   

# %%
import tqdm
import time

def train(
        model,
        train_loader,
        optimizer,
        device,
        num_epochs=3,
        log_interval=1000
):
    model.train()
    model.to(device)
    history = {
        "train_loss": []  # Stocker la loss moyenne pour chaque epoch
    }

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        start_time = time.time()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm.tqdm(train_loader, desc="Batches", disable=True)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Zero gradients to avoid accumulation
            optimizer.zero_grad()
            # Forward pass
            nli_outputs = model(input_ids, attention_mask, labels)
            loss = nli_outputs["loss"]
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            # Accumulate epoch loss
            epoch_loss += loss.item()

            # Display loss at intervals
            if step % log_interval == 0 and step != 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"Batch {step}/{len(train_loader)} | Avg Loss: {avg_loss:.4f}")

        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_epoch_loss)

        # Display epoch completion time
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds | Avg Loss: {avg_epoch_loss:.4f}")

    return history

# %%
history = train(
    nli_camembert,
    train_loader,
    optimizer,
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_epochs=3,
    log_interval=1000
)

# %%


# %%


# %%


# %%


# %%



