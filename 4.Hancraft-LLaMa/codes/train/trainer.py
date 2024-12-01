
import os
import logging
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingArguments:
    output_dir:str=f'training-{datetime.now().strftime("%Y%m%d_%H:%M:%S")}'
    train_batch_size:int=8
    val_batch_size:int=16
    num_train_epoch:int=5
    logging_step:int=1
    eval_step:int=5
    load_best_model_at_end:bool=True
    

class MyDataset(Dataset):
    
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
    


class Trainer:
    
    def __init__(self, 
                 model:nn.Module, 
                 train_dataset:Dataset, 
                 eval_dataset:Dataset, 
                 training_args:TrainingArguments
                 ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        self.trainer_state = {
            'model struture':model.__str__,
            'training_args':training_args.__str__(),
            'states':[]
        }
        self.train_loader = DataLoader(self.train_dataset, 
                            batch_size=self.training_args.train_batch_size, 
                            shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, 
                                 batch_size=self.training_args.val_batch_size, 
                                 shuffle=False)
        
    
    def train(self):
        

        num_steps_per_epoch = len(self.train_loader)
        for epoch in range(self.training_args.num_train_epoch):
            epoch_loss = 0
            
            epoch_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.training_args.num_train_epoch}", unit="batch")
            for step, (input_ids, attention_mask, labels) in enumerate(epoch_iterator, start=1):
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update epoch loss and tqdm bar
                epoch_loss += loss.item()
                epoch_iterator.set_postfix(step=step, loss=loss.item())


        pass
    
    def eval(self):
        pass
    