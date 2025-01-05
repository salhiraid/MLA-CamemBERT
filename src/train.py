from torch.utils.data import DataLoader
from transformers import CamembertTokenizer
from datasets import load_from_disk

from dataset import OscarDataset



dataset_path = r"C:\Users\Napster\Desktop\M2_ISI\MLA\CamemBERT\MLA-CamemBERT\data\oscar.Arrow"
hf_dataset = load_from_disk(dataset_path)

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
torch_dataset = OscarDataset(hf_dataset, tokenizer)
dataloader = DataLoader(torch_dataset, batch_size=16)

for i, batch in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(batch)
    if i >= 1:
        break