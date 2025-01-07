from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer
import torch 

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
        self.vocab_size = len(tokenizer)

    def __len__(self):
        """
        Returns the size of the dataset.
        :return: Number of examples in the dataset
        """
        return len(self.dataset)

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
        non_pad_indices = torch.arange(seq_len)

        # Randomly select 15% of the tokens to mask
        num_to_mask = max(1, int(seq_len * self.mlm_probability))  # At least 1 token
        mask_indices = non_pad_indices[torch.randperm(seq_len)[:num_to_mask]]  # Randomly sample indices

        # Apply the 80/10/10 masking strategy
        # Replace 80% of the selected tokens with <mask>
        num_replace_with_mask = int(0.8 * num_to_mask)
        indices_to_mask = mask_indices[:num_replace_with_mask]
        input_ids[indices_to_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # Replace 10% of the selected tokens with random tokens
        num_replace_with_random = int(0.1 * num_to_mask)
        indices_to_random = mask_indices[num_replace_with_mask:num_replace_with_mask + num_replace_with_random]
        random_tokens = torch.randint(len(self.tokenizer), (len(indices_to_random),), dtype=torch.long)
        input_ids[indices_to_random] = random_tokens

        # Keep 10% of the selected tokens unchanged (handled implicitly as the remaining indices)
        indices_to_unchanged = mask_indices[num_replace_with_mask + num_replace_with_random:]

        # Update the labels: Only compute loss for the selected tokens
        labels[mask_indices] = input_ids[mask_indices]  # Keep the masked tokens in labels
        labels[~torch.isin(torch.arange(seq_len), mask_indices)] = -100  # Ignore others

        return input_ids, labels


    def __getitem__(self, idx):
        """
        Returns an item from the dataset.
        :param idx: Index of the item
        :return: Raw text at the specified index
        """
        assert idx < len(self.dataset), "Index out of range"
        text = self.dataset[idx]['text']
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            #max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        masked_input_ids, labels = self.dynamic_masking(input_ids, attention_mask)  # to check how we verify the ouput
        if self.return_tokens :
            return{
                "input_ids": tokens["input_ids"].squeeze(0),  # Tensor 1D
                "attention_mask": tokens["attention_mask"].squeeze(0),  # Tensor 1D
                "tokens" : self.tokenizer.tokenize(text),
                "text" : text
            }        
        else :
            return{
                "input_ids": tokens["input_ids"].squeeze(0),  # Tensor 1D
                "attention_mask": tokens["attention_mask"].squeeze(0),  # Tensor 1D
                    }