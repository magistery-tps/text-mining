import torch
from torch import nn
import pytorch_common.util as pu
from torch.utils.data import DataLoader 
import logging


class BertModel:
    def __init__(
        self,
        classifier,
        batch_size,
        criterion, 
        optimizer, 
        train_shuffle = True, 
        device        = pu.get_device()
    ):
        self.classifier    = classifier.to(device)
        self.batch_size    = batch_size
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.train_shuffle = train_shuffle
        self.device        = device
    
    def fit(self, train_dataset, val_dataset=None, epochs=10):
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)

        for epoch in range(epochs):
                sw = pu.Stopwatch()
                acc_sum, loss_sum = 0, 0

                for features, target in dataloader:
                    self.classifier.train()

                    target         = target.to(self.device)
                    input_id, mask = self.__get_input(features)

                    output = self.classifier(input_id, mask)
               
                    self.classifier.zero_grad()
                
                    # compute loss and accuracy
                    loss = self.criterion(output, target.long())
                    loss_sum += loss.item()
                    acc_sum  += (output.argmax(dim=1) == target).sum().item()

                    loss.backward()
                    self.optimizer.step()

                tot_train_acc  =  acc_sum  / len(train_dataset)
                tot_train_loss =  loss_sum / len(train_dataset)
                log_message = f'Epoch: {epoch + 1} | Train(loss: {tot_train_loss:.6f}, acc: {tot_train_acc * 100:.2f}%)'

                if val_dataset:
                    val_summary = self.validate(val_dataset)
                    acc_diff = abs(tot_train_acc -  val_summary["acc"])
                    log_message += f' | Val(loss: {val_summary["loss"]:.6f}, acc: {val_summary["acc"] * 100:.2f}%) | acc diff: {acc_diff * 100:.2f}%'

                    resposne_time = sw.elapsed_time()

                logging.info(f'Time: {sw.to_str()} | {log_message}')

        return {'loss': tot_train_loss, 'acc': tot_train_acc}


    def validate(self, dataset):
        dataloader         = DataLoader(dataset, batch_size=self.batch_size)
        acc_sum, loss_sum  = 0, 0
        
        self.classifier.eval()
        with torch.no_grad():
            for features, target in dataloader:
                target         = target.to(self.device)
                input_id, mask = self.__get_input(features)

                output = self.classifier(input_id, mask)

                # compute loss and accuracy
                loss_sum += self.criterion(output, target.long()).item()
                acc_sum  += (output.argmax(dim=1) == target).sum().item()
 
        return {
            'loss': loss_sum / len(dataset), 
            'acc': acc_sum / len(dataset)
        }

    
    def __get_input(self, features):
        mask     = features['attention_mask'].to(self.device)
        input_id = features['input_ids'].squeeze(1).to(self.device)
        return input_id, mask