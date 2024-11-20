from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

cache_dir = "CamemBERT/data"
hf_dataset = load_dataset("oscar-corpus/OSCAR-2201",
                          language="fr",
                          split="train", 
                          cache_dir=cache_dir,
                          trust_remote_code=True)

class OscarDataset(Dataset):
    """
    Wraps Hugging Face dataset into a PyTorch Dataset.
    """
    def __init__(self, hf_dataset):
        """
        Initializes the PyTorch Dataset with the Hugging Face dataset.
        :param hf_dataset: Hugging Face dataset object
        """
        self.data = hf_dataset

    def __len__(self):
        """
        Returns the size of the dataset.
        :return: Number of examples in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns an item from the dataset.
        :param idx: Index of the item
        :return: Raw text at the specified index
        """
        text = self.data[idx]['text']
        return text

torch_dataset = OscarDataset(hf_dataset)
dataloader = DataLoader(torch_dataset, batch_size=16, shuffle=True)

for i, batch in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(batch)
    if i >= 2:
        break