
import os
import logging
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from dataset import CasualLLMDataset


@dataclass
class TrainingArguments:
    output_dir:str=f'training-{datetime.now().strftime("%Y%m%d_%H:%M:%S")}'
    learning_rate:float=5e-4
    train_batch_size:int=8
    val_batch_size:int=16
    num_train_epoch:int=5
    device:str='cuda'
     

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
        
    
    def train(self, 
              train_dataset:CasualLLMDataset=None, 
              eval_dataset:CasualLLMDataset=None):
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_args.learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.model.to(self.training_args.device)
        if train_dataset is None:
            train_dataset = self.train_dataset
        train_dataset.to(self.training_args.device)
        train_loader = DataLoader(train_dataset, 
            batch_size=self.training_args.train_batch_size, 
            shuffle=True)
        num_steps_per_epoch = len(train_loader)
        for epoch in range(self.training_args.num_train_epoch):
            epoch_loss = 0
            
            self.model.train()
            epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.training_args.num_train_epoch}", unit="batch")
            for step, (input_ids, attention_mask, labels) in enumerate(epoch_iterator, start=1):
                
                optimizer.zero_grad()
                
                logits = self.model(input_ids, attention_mask)
                shifted_logits = logits[:, :-1, :]  # Remove last token's prediction
                shifted_labels = labels[:, 1:]      # Remove first token
                loss = loss_fn(
                    shifted_logits.reshape(-1, shifted_logits.size(-1)),  # [batch_size * seq_len, vocab_size]
                    shifted_labels.reshape(-1)                           # [batch_size * seq_len]
                )
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Update epoch loss and tqdm bar
                epoch_loss += loss.item() * labels.size(0)
                epoch_iterator.set_postfix(step=step, loss=loss.item())

            avg_loss = epoch_loss / len(train_loader.dataset)
            perplexity = torch.exp(torch.tensor(avg_loss))
            print(f'[train] average loss: {avg_loss} , perplexity: {perplexity}')
            self.trainer_state['states'].append({'epoch':epoch, 'loss':avg_loss, 'perplexity':perplexity})
            avg_loss_eval, perplexity_eval = self.eval(eval_dataset)
            print(f'[eval] average loss: {avg_loss_eval} , perplexity: {perplexity_eval}')
        return
    

    def eval(self, eval_dataset:CasualLLMDataset=None):
        
        self.model.eval()
        total_loss = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.model.to(self.training_args.device)
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_dataset.to(self.training_args.device)
        eval_loader = DataLoader(eval_dataset, 
                batch_size=self.training_args.val_batch_size, 
                shuffle=False)
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(eval_loader):

                logits = self.model(input_ids, attention_mask)
                shifted_logits = logits[:, :-1, :]  # Remove last token's prediction
                shifted_labels = labels[:, 1:]      # Remove first token
                loss = loss_fn(
                    shifted_logits.reshape(-1, shifted_logits.size(-1)),  # [batch_size * seq_len, vocab_size]
                    shifted_labels.reshape(-1)                           # [batch_size * seq_len]
                )
                # Track loss
                total_loss += loss.item() * labels.size(0)  # Scale back up by batch size
            avg_loss = total_loss / len(eval_loader.dataset)
            perplexity = torch.exp(torch.tensor(avg_loss))
        return avg_loss, perplexity

    