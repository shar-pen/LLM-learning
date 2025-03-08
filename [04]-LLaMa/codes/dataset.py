import torch
from torch.utils.data import DataLoader, Dataset


class CasualLLMDataset(Dataset):
    
    def __init__(self, input_ids, attention_mask, labels) -> None:
        super().__init__()
        
        assert len(input_ids) == len(attention_mask) == len(labels)
        self.input_ids = torch.tensor(input_ids)
        self.attention_mask = torch.tensor(attention_mask)
        self.labels = torch.tensor(labels)
        
    def to(self, device='cuda'):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.labels = self.labels.to(device)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]