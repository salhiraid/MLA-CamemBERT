import torch
from torch import nn, Tensor
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader

from transformers import CamembertModel, CamembertTokenizer, CamembertConfig
from datasets import load_dataset

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
    
import pytorch_lightning as pl
from torchmetrics import Accuracy

class NLI(pl.LightningModule):
    def __init__(self, 
                 model: NLIFinetuningModel, 
                 lr: float = 5e-5, 
                 warmup_steps: int = 0, 
                 total_steps: int = 10000):
        """
        Initialize the NLI model.
        :param model: Instance of the NLIFinetuningModel.
        """
        super(NLI, self).__init__()
        self.model = model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_index):
        """
        Training step for the model.
        """
        input_ids, attention_mask, labels = batch
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
        input_ids, attention_mask, labels = batch
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