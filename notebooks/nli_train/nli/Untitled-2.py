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
data_path = "../../../data/XNLI"
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

        self.hidden_size = base_model.get_hidden_size()
        self.num_labels = num_labels

        self.nli_head = nn.Linear(self.hidden_size, num_labels)
        # self.nli_head = NLIHead(base_model.get_hidden_size(), num_labels)

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
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from pytorch_lightning import Trainer

class NLI(pl.LightningModule):
    def __init__(self, model: NLIFinetuningModel, lr: float = 5e-5, warmup_steps: int = 0, total_steps: int = 10000):
        """
        Initialize the NLI model with PyTorch Lightning.
        :param model: Instance of the NLIFinetuningModel.
        :param lr: Learning rate for the optimizer.
        :param warmup_steps: Number of warmup steps for the scheduler.
        :param total_steps: Total steps for linear warmup scheduler.
        """
        super(NLI, self).__init__()
        self.model = model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=3)
        self.val_acc = Accuracy(task="multiclass", num_classes=3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_index):
        """
        Training step for the model.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        
        outputs = self.model(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        # Compute accuracy
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, idx):
        """
        Validation step for the model.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = self.model(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        # Compute accuracy
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)

        # Log loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Aggregate metrics at the end of each validation epoch.
        """
        # Log global validation accuracy
        self.log('val_accuracy', self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()  # Reset accuracy for the next epoch

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.lr, 
            total_steps=self.total_steps, 
            pct_start=0.1, 
            anneal_strategy='linear'
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def forward(self, batch):
        """
        Forward pass for inference.
        """
        input_ids, attention_mask, _ = batch
        outputs = self.model(input_ids, attention_mask)
        return outputs["logits"]

# %%
model_path = "../../../models/oscar_4gb"

# 1. Load the base model
base_camembert = CamembertModel.from_pretrained(model_path)
# 2. Load the NLI model :
nli_camembert = NLIFinetuningModel(base_model=CamemBERTBaseModel(model_path, trainable=True), 
                                   num_labels=3)

nb_epochs = 3
nb_steps_per_epoch = len(train_loader)

pl_camembert = NLI(model=nli_camembert, total_steps=nb_epochs*nb_steps_per_epoch)

# %%
xnli_train_dataset = XNLIDataset(split="train", language="fr", cache_directory=data_path, max_length=256)
xnli_val_dataset = XNLIDataset(split="validation", language="fr", cache_directory=data_path, max_length=256)

train_loader = DataLoader(xnli_train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(xnli_val_dataset, batch_size=8, shuffle=False)

# %%
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


logger = TensorBoardLogger("my_logs", name="my_experiment")

# Configurer le checkpoint
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",         # Sauvegarder le modèle basé sur la val_loss
    dirpath="checkpoints/",     # Répertoire où sauvegarder les modèles
    filename="nli-{epoch:02d}-{val_loss:.2f}",  # Nom des fichiers
    save_top_k=2,               # Sauvegarder le meilleur modèle uniquement
    mode="min"                  # Minimiser val_loss
)

# Configuration du Trainer
trainer = Trainer(
    max_epochs=3,                # Nombre d'epochs
    logger=logger,               # Logger pour TensorBoard
    callbacks=[checkpoint_callback],  # Callback pour le checkpoint
    accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Utilise le GPU si disponible
    devices=1,                   # Nombre de GPUs à utiliser
    log_every_n_steps=10,        # Fréquence d'affichage des logs
    precision=16,                # Utilisation du mixed precision pour accélérer l'entraînement (facultatif)
)

# %%
trainer.fit(pl_camembert, train_loader, val_loader)

# %%


# %% [markdown]
# - 1. il faut aussi avoir une metric ainsi qu'une visualisation
# - 2. regler le truc dans jupyter lab surtout avec les fichiers de data (faire un git clone)


