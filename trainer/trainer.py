import os
import time
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from dataset import CrosswordClueAnswersDataset
from models import SimpleCrosswordModel
from training import Trainer
from trainingstats import TrainingReporter

PADDING_TOKEN_INDEX = 0
PAD_TO_SIZE = 45

def run(args):
    device = args.device
    if device == 'auto':        
        #TODO: Add support for cuda
        # attempt to run on mps - will do work on the GPU for MacOS
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f'Running on:', device)

    # load data, split datasets, build vocabs
    #TODO: Parameterize this
    dataset = CrosswordClueAnswersDataset("cleaned_data/clean_3.csv")
    train_size = int(0.8 * len(dataset))
    dev_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - dev_size
    g = torch.Generator().manual_seed(42) # this manual_seed is important to ensure that we consistently split the dataset
    train_dataset, test_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, dev_size], generator=g)

    # build vocab onlt off of training data (for now...)
    tokenizer = get_tokenizer('basic_english')

    clues_iter = map(lambda data: tokenizer(data[1]), train_dataset)
    answers_iter = map(lambda data: tokenizer(data[0]), train_dataset)
    
    clues_vocab = build_vocab_from_iterator(clues_iter, specials=['<pad>', '<unk>'])
    clues_vocab.set_default_index(1)

    answers_vocab = build_vocab_from_iterator(answers_iter, specials=['<unk>'])
    answers_vocab.set_default_index(0)

    print(f'{len(dataset)=}\n{len(train_dataset)=}\n{len(test_dataset)=}\n{len(dev_dataset)=}')
    print(f'{len(answers_vocab)=}\n{len(clues_vocab)=}')

    def collate_batch(batch):
        answer_list, clue_list = [], []

        for (answer, clue) in batch:
            clue_indicies = clues_vocab(tokenizer(clue))
            clue_indicies += [PADDING_TOKEN_INDEX] * (PAD_TO_SIZE - len(clue_indicies))
            clue_list.append(clue_indicies)

            answer_list.append(answers_vocab([answer])[0])

        answer_list = torch.tensor(answer_list).to(device)
        clue_list = torch.tensor(clue_list).to(device)

        return answer_list, clue_list

    # shuffle the training dataloader so we go through different batches each time
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch) 

    print(f'{len(train_dataloader)=}\n{len(dev_dataloader)=}')

    model = SimpleCrosswordModel(
        vocab_size=len(clues_vocab),
        embed_dim=args.embedding_dimensions,
        input_size=PAD_TO_SIZE,
        hidden_size=args.hidden_layer_size,
        output_size=len(answers_vocab),
        device=device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_dir = os.path.join('training_results', args.output_folder)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    print(f'Outputting to: {train_dir}')

    log_interval = int(len(train_dataloader) / 5)
    reporter = TrainingReporter(train_dir, log_interval)
    training = Trainer(model=model, criterion=criterion, optimizer=optimizer, reporter=reporter, output_dir=train_dir)
    training.start(num_epochs=args.num_epochs, train_dataloader=train_dataloader, test_dataloader=dev_dataloader)
