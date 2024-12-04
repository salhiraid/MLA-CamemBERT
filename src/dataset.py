import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer
from datasets import load_dataset, load_from_disk


class OscarDataset(Dataset):
    """
    Wraps Hugging Face dataset into a PyTorch Dataset.
    """
    def __init__(self, hf_dataset, tokenizer, max_length=512, return_tokens=False):
        """
        Initializes the PyTorch Dataset with the Hugging Face dataset.

        :param hf_dataset: Dataset au format Arrow (Hugging Face Dataset)
        :param tokenizer: Tokenizer (ex. CamembertTokenizer)
        :param max_length: Longueur maximale des séquences après tokenisation
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tokens = return_tokens
        self.mlm_probability = 0.15

    def __len__(self):
            """
            Returns the size of the dataset.
            :return: Number of examples in the dataset
            """
            return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns an item from the dataset.
        """
        assert idx < len(self.dataset), "Index out of range"
        text = self.dataset[idx]['text']
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Apply dynamic masking
        masked_input_ids, labels = self.dynamic_masking(input_ids, attention_mask)

        return {
            "masked_input_ids": masked_input_ids,  # Masked input IDs
            "attention_mask": attention_mask,
            "labels": labels,  # Labels for MLM loss
        }

    def dynamic_masking(self, input_ids, attention_mask):
        """
        Applies dynamic masking for MLM using random index selection.
    
        :param input_ids: Tensor of tokenized input IDs
        :param attention_mask: Tensor of attention mask to identify non-padded tokens
        :return: Masked input IDs and labels for MLM loss computation
        """
        seq_len = attention_mask.sum().item()  # Find the length of the non-padded sequence
        labels = input_ids.clone()  # Clone input_ids to create labels
        
        # Get the indices of non-padding tokens
        non_pad_indices = torch.where(attention_mask == 1)[0]  # Indices of non-padded tokens
    
        # Randomly select 15% of the tokens to mask
        num_to_mask = max(1, int(seq_len * self.mlm_probability))  # At least 1 token
        mask_indices = non_pad_indices[torch.randperm(len(non_pad_indices))[:num_to_mask]]  # Randomly sample indices
    
        # Apply the 80/10/10 masking strategy
        # Replace 80% of the selected tokens with <mask>
        num_replace_with_mask = int(0.8 * num_to_mask)
        indices_to_mask = mask_indices[:num_replace_with_mask]
        input_ids[indices_to_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
    
        # Replace 10% of the selected tokens with random tokens
        num_replace_with_random = int(0.1 * num_to_mask)
        indices_to_random = mask_indices[num_replace_with_mask:num_replace_with_mask + num_replace_with_random]
        random_tokens = torch.randint(self.tokenizer.vocab_size, (len(indices_to_random),), dtype=torch.long)
        input_ids[indices_to_random] = random_tokens
    
        # Keep 10% of the selected tokens unchanged (handled implicitly as the remaining indices)
        # For the remaining 10%, no action needed since input_ids is already unchanged
    
        # Update the labels
        # Set labels to -100 for tokens that are not masked (so they are ignored in loss computation)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[mask_indices] = True
        labels[~mask] = -100  # Only compute loss for masked tokens
    
        return input_ids, labels
