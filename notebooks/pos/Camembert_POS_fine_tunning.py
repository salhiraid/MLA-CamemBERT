import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer,
    CamembertForTokenClassification,
    AdamW,
    DataCollatorForTokenClassification
)
from datasets import load_dataset

import matplotlib.pyplot as plt
import os

############################################################################
# 1) Load the UD French Sequoia dataset
############################################################################
dataset = load_dataset(
    "universal_dependencies",
    "fr_spoken",
    trust_remote_code=True
)

print("Dataset splits:\n", dataset)  # train, validation, test

############################################################################
# 2) Inspect the labels (UPOS) and set up id2label/label2id for readability
############################################################################
upos_feature = dataset["train"].features["upos"]
all_label_strings = upos_feature.feature.names
num_labels = len(all_label_strings)

id2label = {i: label for i, label in enumerate(all_label_strings)}
label2id = {label: i for i, label in enumerate(all_label_strings)}

print("\nUD labels (string) =>", all_label_strings)
print("Number of distinct UPOS labels:", num_labels)

############################################################################
# 3) Define the CamemBERT tokenizer
############################################################################
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

############################################################################
# 4) Tokenize and align labels function (manual, batched version)
############################################################################
def tokenize_and_align_labels(examples):
    """
    examples["tokens"]: list of lists (one list per sentence)
    examples["upos"]:   list of lists (each sub-list is a sequence of integer labels)
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True
    )

    all_labels = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # ignore special tokens
            else:
                # The UPOS here is already an integer (e.g., 0..17)
                upos_id = examples["upos"][i][word_id]
                labels.append(upos_id)
        all_labels.append(labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

############################################################################
# 5) Apply the function to the dataset
############################################################################
tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True
)

############################################################################
# 6) Convert to PyTorch format
############################################################################
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

print("\nSample tokenized entry from train set:")
print(tokenized_dataset["train"][0])

############################################################################
# 7) Define the model, set readable label mappings
############################################################################
model = CamembertForTokenClassification.from_pretrained(
    "camembert-base",
    num_labels=num_labels
)
model.config.id2label = id2label
model.config.label2id = label2id

############################################################################
# 8) Create DataLoaders (batching) for train, validation, and test
############################################################################
train_dataset = tokenized_dataset["train"]
val_dataset   = tokenized_dataset["validation"]
test_dataset  = tokenized_dataset["test"]

batch_size = 16

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),  # shuffle
    batch_size=batch_size,
    collate_fn=data_collator
)

val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),  # no shuffle
    batch_size=batch_size,
    collate_fn=data_collator
)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=batch_size,
    collate_fn=data_collator
)

############################################################################
# 9) Define optimizer, epochs, etc.
############################################################################
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

############################################################################
# 10) Training Loop with validation accuracy
#     We'll record train_loss, val_loss, val_accuracy each epoch
############################################################################
train_losses = []
val_losses   = []
val_accuracies = []

for epoch in range(num_epochs):
    # -------------------------
    # TRAINING
    # -------------------------
    model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        # Move batch to the same device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # -------------------------
    # VALIDATION (Loss + Accuracy)
    # -------------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            val_loss += outputs.loss.item()

            # Predictions
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            labels = batch["labels"]

            # Only consider positions where label != -100
            mask = labels != -100
            val_correct += (preds[mask] == labels[mask]).sum().item()
            val_total   += mask.sum().item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = val_correct / val_total if val_total > 0 else 0.0

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_accuracy:.4f}")

############################################################################
# 11) Evaluate on Test Set (Loss + Accuracy)
############################################################################
model.eval()
test_loss = 0.0
test_correct = 0
test_total   = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        test_loss += outputs.loss.item()

        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]

        mask = labels != -100
        test_correct += (preds[mask] == labels[mask]).sum().item()
        test_total   += mask.sum().item()

avg_test_loss = test_loss / len(test_dataloader)
test_accuracy = test_correct / test_total if test_total > 0 else 0.0

print(f"\nFinal Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

############################################################################
# 12) Save the Model + Tokenizer
############################################################################
save_directory = "./my_camembert_pos_model_fr_ftb"
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)    # saves config + weights
tokenizer.save_pretrained(save_directory)  # saves the tokenizer files

print(f"\nModel and tokenizer saved to: {save_directory}")

############################################################################
# 13) Plot Training & Validation Loss
############################################################################
# We'll use matplotlib to plot the losses across epochs.
epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, val_losses,   label='Val Loss', marker='o')
plt.title("Training & Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")  # Save the figure to a file
plt.show()

# Optionally, you can plot validation accuracy as well:
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, val_accuracies, label='Val Accuracy', marker='o', color='green')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])  # accuracy goes from 0 to 1
plt.grid(True)
plt.legend()
plt.savefig("accuracy_plot.png")
plt.show()
