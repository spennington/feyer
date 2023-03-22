import json
import os
import time
import torch

class TrainingReporter:
    
    def __init__(self, output_dir, batch_log_interval) -> None:
        self.batch_log_interval = batch_log_interval
        self.__reset_batch_stats()

        # stats to track for each epoch
        self.test_accu_pcts, self.test_losses = [], []
        self.train_accu_pcts, self.train_losses = [], []
        self.elapsed_times = []

        # setup directories and files for output
        self.results_filename = os.path.join(output_dir, 'training-results.json')

    def __reset_batch_stats(self):
        # stats to track for each batch
        self.log_interval_batch_count = 0
        self.log_interval_count = 0
        self.log_interval_acc = 0
        self.log_interval_loss = 0
        self.log_interval_elapsed = 0

    def batch_complete(self, epoch: int, batch_index: int, num_batches: int, batch_size: int, accuracy: float, loss: float, elapsed: float):
        self.log_interval_batch_count += 1
        self.log_interval_count += batch_size
        self.log_interval_acc += accuracy
        self.log_interval_loss += loss
        self.log_interval_elapsed += elapsed

        if batch_index != 0 and batch_index % self.batch_log_interval == 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f} | loss {:8.7f} | time {:5.2f}s '.format(epoch, batch_index, num_batches,
                                              self.log_interval_acc / self.log_interval_count,
                                              self.log_interval_loss / self.log_interval_batch_count,
                                              self.log_interval_elapsed))
            
            self.__reset_batch_stats()

    def epoch_complete(self, epoch: int, train_accuracy: float, train_loss: float, test_accuracy: float, test_loss: float, elapsed: float):
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
                'dev accuracy {:8.3f} | dev loss {:8.7f} '.format(epoch,
                                                elapsed,
                                                test_accuracy, test_loss))
        print('-' * 59)

        self.test_accu_pcts.append(test_accuracy)
        self.test_losses.append(test_loss)
        self.train_accu_pcts.append(train_accuracy)
        self.train_losses.append(train_loss)
        self.elapsed_times.append(elapsed)

        #TODO: log hyperparams somewhere and previous model
        training_dict = {
            #'hyperparameters': hyperparameters,
            'dev_accu_pcts': self.test_accu_pcts,
            'dev_losses': self.test_losses,
            'train_accu_pcts': self.train_accu_pcts,
            'train_losses': self.train_losses,
            'elapsed_times': self.elapsed_times,
            #'model': model_filename,
            #'previous_model': previous_model
        }

        with open(self.results_filename, 'w') as file:
                file.write(json.dumps(training_dict))

        self.__reset_batch_stats()