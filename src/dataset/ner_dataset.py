from datasets import load_dataset
from torch.utils.data import  Dataset
import torch

class NERDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128, label_all_tokens=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens

    def tokenize_and_align_labels(self, tokens, ner_tags):
        tokenized_inputs = self.tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            padding='max_length',
            max_length=self.max_length
        )
        labels = []
        for i, label in enumerate(ner_tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        # Convert all outputs to tensors
        tokenized_inputs = {key: torch.tensor(val, dtype=torch.long) for key, val in tokenized_inputs.items()}
        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

        return tokenized_inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        tokenized_data = self.tokenize_and_align_labels(
            [data['tokens']], [data['ner_tags']]
        )

        return {
            'input_ids': tokenized_data['input_ids'].squeeze(),
            'attention_mask': tokenized_data['attention_mask'].squeeze(),
            'labels': torch.tensor(tokenized_data['labels'][0], dtype=torch.long),
        }

