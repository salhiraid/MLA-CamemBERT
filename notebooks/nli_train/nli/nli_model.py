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
    

