import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CamembertTokenizer

from ..dataset import XNLIDataset
from ..models.nli_model import NLIFinetuningModel



def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one training epoch.

    Args:
        model: Model to train.
        loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        device: Device to use (e.g., "cuda").

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates the model on validation data.

    Args:
        model: Model to evaluate.
        loader: DataLoader for validation data.
        device: Device to use.

    Returns:
        tuple: Average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            running_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return running_loss / len(loader), accuracy


# Load tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Load datasets
train_dataset = XNLIDataset(split="train", tokenizer=tokenizer, language="fr")
val_dataset = XNLIDataset(split="validation", tokenizer=tokenizer, language="fr")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model
model_path = "../../models/4gb_oscar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NLIFinetuningModel(base_model_path=model_path, num_labels=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(5):
    print(f"Epoch {epoch + 1}")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")