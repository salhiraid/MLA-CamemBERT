import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class XNLIDataset(Dataset):
    """
    PyTorch dataset for the XNLI dataset.
    """

    def __init__(self, split="train", language="fr", tokenizer=None, cache_directory="../data/xnli", max_length=128):
        """
        Initializes the dataset.

        Args:
            split (str): Dataset split ("train", "test", "validation").
            language (str): Target language.
            tokenizer: Tokenizer instance.
            cache_directory (str): Cache directory for downloaded data.
            max_length (int): Maximum length for padding/truncation.
        """
        self.split = split
        self.language = language
        self.tokenizer = tokenizer
        self.cache_directory = cache_directory
        self.max_length = max_length

        self.data = load_dataset(
            "facebook/xnli",
            name=self.language,
            cache_dir=self.cache_directory
        )[self.split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        inputs = self.tokenizer(
            example["premise"],
            example["hypothesis"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["label"] = torch.tensor(example["label"], dtype=torch.long)
        return inputs
