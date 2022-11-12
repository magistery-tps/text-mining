import torch
from torch import nn
import pytorch_common.util as pu
from torch.utils.data import DataLoader
import logging
import numpy as np
import logging
from IPython.display import display
from .metrics import plot_metrics
from sklearn.metrics import classification_report


class EvaluationSumamry:
    def __init__(self, predictions, targets, loss, accuracy):
        self.predictions    = predictions
        self.targets        = targets
        self.loss           = loss
        self.accuracy       = accuracy

    def plot_sample_metrics(self, index, figuresize=(12, 12)):        
        plot_metrics(self.targets[index], self.predictions[index], figuresize=figuresize)

    def plot_metrics(self, figuresize=(25, 25), label_by_class=None):
        targets     = np.concatenate(self.targets)
        predictions = np.concatenate(self.predictions)

        if label_by_class:
            targets      = [label_by_class[c] for c in targets]
            predictions  = [label_by_class[c] for c in predictions]

        plot_metrics(targets, predictions, figuresize)

    def show(self):
        print(f'Accuracy: {self.accuracy*100:.2f}%, Loss: {self.loss:.6f}')

    def predicted_classes_by(self, less_than_f1_score, label_by_class):
        targets             = [label_by_class[c] for c in np.concatenate(self.targets)]
        predictions         = [label_by_class[c] for c in np.concatenate(self.predictions)]

        report = classification_report(targets,  predictions, output_dict=True)

        classess = []
        for k, v in report.items():
            if type(v) == dict  and 'f1-score' in v and v['f1-score'] < less_than_f1_score:
                classess.append(k)
        return classess


class ModelTrainer:
    def __init__(
        self,
        classifier,
        batch_size,
        criterion,
        train_shuffle = True,
        device        = pu.get_device()
    ):
        self.classifier    = classifier.to(device)
        self.batch_size    = batch_size
        self.criterion     = criterion
        self.train_shuffle = train_shuffle
        self.device        = device

    def fit(self, train_dataset, optimizer, val_dataset=None, epochs=10):
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
                    optimizer.step()

                tot_train_acc  =  acc_sum  / len(train_dataset)
                tot_train_loss =  loss_sum / len(train_dataset)
                log_message = f'Epoch: {epoch + 1} | Train(loss: {tot_train_loss:.6f}, acc: {tot_train_acc * 100:.2f}%)'

                if val_dataset:
                    summary = self.validate(val_dataset)
                    acc_diff = abs(tot_train_acc -  summary.accuracy)
                    log_message += f' | Val(loss: {summary.loss:.6f}, acc: {summary.accuracy * 100:.2f}%) | acc diff: {acc_diff * 100:.2f}%'

                    resposne_time = sw.elapsed_time()

                logging.info(f'Time: {sw.to_str()} | {log_message}')

        return {'loss': tot_train_loss, 'acc': tot_train_acc}


    def validate(self, dataset):
        dataloader         = DataLoader(dataset, batch_size=self.batch_size)
        acc_sum, loss_sum  = 0, 0

        predictions = []
        targets     = []

        self.classifier.eval()
        with torch.no_grad():
            for features, target in dataloader:
                targets.append(target.cpu().numpy())
                target         = target.to(self.device)
                input_id, mask = self.__get_input(features)

                output = self.classifier(input_id, mask)
                predictions.append(output.argmax(dim=1).cpu().numpy())

                # compute loss and accuracy
                loss_sum += self.criterion(output, target.long()).item()
                acc_sum  += (output.argmax(dim=1) == target).sum().item()

        return EvaluationSumamry(
            predictions,
            targets,
            loss     = loss_sum / len(dataset),
            accuracy = acc_sum / len(dataset)
        )


    def __get_input(self, features):
        mask     = features['attention_mask'].to(self.device)
        input_id = features['input_ids'].squeeze(1).to(self.device)
        return input_id, mask