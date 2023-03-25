import os
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer

from trainer.crossword.models import SimpleCrosswordModel
from trainer.crossword.trainingstats import TrainingReporter

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

class CharacterTrainer:

    def __init__(self, model: torch.nn.Module, criterion: CrossEntropyLoss, optimizer: Optimizer, reporter: TrainingReporter, output_dir: str) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.reporter = reporter

    def evaluate(self, dataloader, itos):
        """
        Evaluate the model against a dataset
        Returns a tuple of (acurate_pct, loss)
        """
        self.model.eval()
        running_loss, running_acc, running_count = 0, 0, 0

        with torch.no_grad():
            for idx, (inputs, target_outputs, input_lengths, answer_lengths, clues, answers) in enumerate(dataloader):
                output, _ = self.model(inputs)
                reshaped_outputs = output.reshape(-1, output.size(2))
                reshaped_target_outputs = target_outputs.reshape(-1)
                loss = self.criterion(reshaped_outputs, reshaped_target_outputs).item()

                predicted_tokens = output.argmax(2)
                #print(f'{predicted_tokens.shape=}')
                #print(f'{predicted_tokens[:,-3:].shape=}')
                for i, answer_length in enumerate(answer_lengths):
                    #print(f'{clues[i]=}')
                    #print(f'{answers[i]=}')
                    # <BOC> a b c d <EOC> <4> e f g h <EOA>
                    actual_answer_start = input_lengths[i] - answer_length
                    # predicted answer starts one sooner since we'd expect an <EOA>
                    predicted_answer_start = actual_answer_start - 1
                    predicted_answer = predicted_tokens[i, predicted_answer_start:predicted_answer_start+answer_length:]
                    actual_answer = torch.argmax(inputs[i, actual_answer_start:actual_answer_start+answer_length], dim=1)
                    if torch.equal(actual_answer, predicted_answer):
                        running_acc += 1
                        #print(clues[i], answers[i], list(map(lambda x: itos[x], predicted_answer)))
                    #print(f'{predicted_tokens=}')
                    #print(f'{predicted_answer=}')
                    #print(f'{actual_answer=}')

                #running_acc += (predicted_answer.argmax(1) == answer).sum().item()
                running_count += target_outputs.size(0)
                running_loss += loss
        return running_acc / running_count, running_loss / len(dataloader)

    def train(self, dataloader, epoch, itos):
        """
        Train the model against a dataset
        Returns a tuple of (acurate_pct, loss)
        """
        self.model.train()

        running_loss, running_acc, running_count = 0, 0, 0

        for idx, (inputs, target_outputs, input_lengths, answer_lengths, clues, answers) in enumerate(dataloader):
            batch_start_time = time.time()
            self.optimizer.zero_grad()
            #print(f'{target_outputs=}')
            #print(f'{inputs.shape=}{target_outputs.shape=}')
            #print(f'{packed_inputs.shape=}')
            output, _ = self.model(inputs) # (batch_size, pad_length, vocab_size)
            #print(f'{output.shape=}')
            reshaped_outputs = output.reshape(-1, output.size(2))
            #print(f'{reshaped_outputs.shape=}')
            reshaped_target_outputs = target_outputs.reshape(-1)
            #print(f'{reshaped_target_outputs.shape=}')

            #TODO: only compare first input_lengths tokens
            #print(f'{target_outputs=}')
            loss = self.criterion(reshaped_outputs, reshaped_target_outputs)
            loss.backward()
            self.optimizer.step()

            # take the last answer_lenghts values from the outputs
            #print(f'{answer_lengths.dtype=}')
            #last_n_outputs = output[:,-answer_lengths:,:]
            #print(last_n_outputs.shape)

            #TODO: this is too slow
            #predicted_tokens = output.argmax(2)
            #print(f'{predicted_tokens.shape=}')
            #print(f'{predicted_tokens[:,-3:].shape=}')
            batch_acc = 0
            #for i, answer_length in enumerate(answer_lengths):
                #print(f'{clues[i]=}')
                #print(f'{answers[i]=}')
                # <BOC> a b c d <EOC> <4> e f g h <EOA>
            #    actual_answer_start = input_lengths[i] - answer_length
                # predicted answer starts one sooner since we'd expect an <EOA>
            #    predicted_answer_start = actual_answer_start - 1
            #    predicted_answer = predicted_tokens[i, predicted_answer_start:predicted_answer_start+answer_length:]
            #    actual_answer = torch.argmax(inputs[i, actual_answer_start:actual_answer_start+answer_length], dim=1)
            #    if torch.equal(actual_answer, predicted_answer):
            #        batch_acc += 1
                #print(f'{predicted_tokens=}')
                #print(f'{predicted_answer=}')
                #print(f'{actual_answer=}')

            batch_size = target_outputs.size(0)

            running_count += batch_size
            running_loss += loss.item()
            running_acc += batch_acc

            elapsed = time.time() - batch_start_time
            self.reporter.batch_complete(epoch, idx, len(dataloader), batch_size, batch_acc, loss.item(), elapsed)

        return running_acc / running_count, running_loss / len(dataloader)

    def start(self, num_epochs, train_dataloader, test_dataloader, vocab):
        itos = vocab.get_itos()
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            train_accu_pct, train_loss = self.train(train_dataloader, epoch, itos)
            test_accu_pct, test_loss = self.evaluate(test_dataloader, itos)

            elapsed = time.time() - epoch_start_time
            model_filename = os.path.join(self.output_dir, 'model-full-epoch-' + str(epoch) + '.pt')
            torch.save(self.model, model_filename)

            self.reporter.epoch_complete(epoch, train_accu_pct, train_loss, test_accu_pct, test_loss, elapsed)