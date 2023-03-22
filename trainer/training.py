import os
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer

from models import SimpleCrosswordModel
from trainingstats import TrainingReporter

class Trainer:

    def __init__(self, model: SimpleCrosswordModel, criterion: CrossEntropyLoss, optimizer: Optimizer, reporter: TrainingReporter, output_dir: str) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.reporter = reporter

    def evaluate(self, dataloader):
        """
        Evaluate the model against a dataset
        Returns a tuple of (acurate_pct, loss)
        """
        self.model.eval()
        running_loss, running_acc, running_count = 0, 0, 0

        with torch.no_grad():
            for idx, (answer, clue) in enumerate(dataloader):
                predicted_answer = self.model(clue)
                loss = self.criterion(predicted_answer, answer).item()
                running_acc += (predicted_answer.argmax(1) == answer).sum().item()
                running_count += answer.size(0)
                running_loss += loss
        return running_acc / running_count, running_loss / len(dataloader)

    def train(self, dataloader, epoch):
        """
        Train the model against a dataset
        Returns a tuple of (acurate_pct, loss)
        """
        self.model.train()

        running_loss, running_acc, running_count = 0, 0, 0

        for idx, (answer, clue) in enumerate(dataloader):
            batch_start_time = time.time()
            self.optimizer.zero_grad()
            predicted_answer = self.model(clue)
            loss = self.criterion(predicted_answer, answer)
            loss.backward()
            self.optimizer.step()
            
            batch_acc = (predicted_answer.argmax(1) == answer).sum().item()
            batch_size = answer.size(0)

            running_count += batch_size
            running_loss += loss.item()
            running_acc += batch_acc

            elapsed = time.time() - batch_start_time
            self.reporter.batch_complete(epoch, idx, len(dataloader), batch_size, batch_acc, loss.item(), elapsed)

        return running_acc / running_count, running_loss / len(dataloader)

    def start(self, num_epochs, train_dataloader, test_dataloader):    
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            train_accu_pct, train_loss = self.train(train_dataloader, epoch)
            test_accu_pct, test_loss = self.evaluate(test_dataloader)

            elapsed = time.time() - epoch_start_time
            model_filename = os.path.join(self.output_dir, 'model-epoch-' + str(epoch) + '.pt')
            torch.save(self.model.state_dict(), model_filename)
            self.reporter.epoch_complete(epoch, train_accu_pct, train_loss, test_accu_pct, test_loss, elapsed)