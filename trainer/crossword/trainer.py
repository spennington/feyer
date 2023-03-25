import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from trainer.crossword.dataset import CrosswordClueAnswersDataset
from trainer.crossword.models import SimpleCrosswordModel
from trainer.crossword.models import CharacterRNN
from trainer.crossword.training import CharacterTrainer, Trainer
from trainer.crossword.trainingstats import TrainingReporter

def build_character_model(dataset, device, args):
    token_iter = map(lambda data: list(data[1]) + list(data[0]), dataset)
    max_answer_length = max(map(lambda data: len(data[0]), dataset))
    print(f'{max_answer_length=}')
    answer_tokens = []
    for i in range(1, max_answer_length + 1):
        answer_tokens.append(f'<{str(i)}>')
    # padding at 0 is important
    specials = ['<PAD>', '<BOC>', '<EOC>', '<EOA>']
    specials.extend(answer_tokens)
    vocab = build_vocab_from_iterator(token_iter, specials=specials)
    print(f'{len(vocab)=}')
    print(f'{vocab.get_itos()=}')

    def transform_tokens(clue, answer):
        tokens = ['<BOC>']
        tokens.extend(list(clue))
        tokens.append('<EOC>')
        tokens.append('<' + str(len(answer)) + '>')
        tokens.extend(list(answer))
        tokens.append('<EOA>')
        return tokens

    def collate_batch(batch):
        inputs, outputs, input_lengths, answer_lengths, clues, answers = [], [], [], [], [], []

        for (answer, clue) in batch:
            tokens = transform_tokens(clue=clue, answer=answer)
            token_indicies = torch.tensor(vocab(tokens))
            one_hots = F.one_hot(token_indicies, len(vocab))
            inputs.append(one_hots[:-1].float()) # skip last token, float so mps works
            clue_length_with_specials = len(clue) + 3
            output = []
            output.extend([0] * (clue_length_with_specials - 1)) # skip first token since that's not an output
            output.extend(token_indicies[clue_length_with_specials:]) # skip clue tokens since we just treat those outputs as padding
            #print(tokens, output)
            outputs.append(torch.tensor(output))
            input_lengths.append(len(token_indicies) - 1)
            answer_lengths.append(len(answer))
            clues.append(clue)
            answers.append(answer)

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True).to(device)
        outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True).to(device)
        #answer_lengths = torch.tensor(answer_lengths).int()
        #input_lengths = torch.tensor(input_lengths)

        return inputs, outputs, input_lengths, answer_lengths, clues, answers

    model = CharacterRNN(vocab_size=len(vocab), hidden_size=args.hidden_layer_size, device=device, num_layers=args.hidden_depth)

    return model, collate_batch, vocab

def build_token_model(train_dataset, device, args):
    PADDING_TOKEN_INDEX = 0
    PAD_TO_SIZE = 11

    # build vocab onlt off of training data (for now...)
    tokenizer = get_tokenizer('basic_english')

    clues_iter = map(lambda data: tokenizer(data[1]), train_dataset)
    answers_iter = map(lambda data: tokenizer(data[0]), train_dataset)

    clues_vocab = build_vocab_from_iterator(clues_iter, specials=['<pad>', '<unk>', '<1>', '<2>', '<3>', '<4>', '<5>', '<6>', '<7>', '<8>', '<9>', '<10>', '<11>', '<12>', '<13>', '<14>', '<15>', '<16>', '<17>', '<18>', '<19>', '<20>', '<21>', '<22>'])
    clues_vocab.set_default_index(1)

    answers_vocab = build_vocab_from_iterator(answers_iter, specials=['<unk>'])
    answers_vocab.set_default_index(0)

    print(f'{len(answers_vocab)=}\n{len(clues_vocab)=}')

    model = SimpleCrosswordModel(
        vocab_size=len(clues_vocab),
        embed_dim=args.embedding_dimensions,
        input_size=PAD_TO_SIZE,
        hidden_size=args.hidden_layer_size,
        output_size=len(answers_vocab),
        device=device,
        hidden_depth=args.hidden_depth)

    # model = RecurrentCrosswordModel(
    #     vocab_size=len(clues_vocab),
    #     output_size=len(answers_vocab),
    #     embed_dim=args.embedding_dimensions,
    #     hidden_size=args.hidden_layer_size,
    #     hidden_depth=args.hidden_depth,
    #     device=device
    # )

    def collate_batch(batch):
        answer_list, clue_list = [], []

        for (answer, clue) in batch:
            answer_length = len(answer)
            length_token = '<' + str(answer_length) + '>'
            tokens = tokenizer(clue)
            tokens.insert(0, length_token)
            clue_indicies = clues_vocab(tokens)
            pad_len = PAD_TO_SIZE - len(clue_indicies)
            if pad_len < 0:
                raise Exception('pad_len < 0')
            clue_indicies += [PADDING_TOKEN_INDEX] * (pad_len)
            clue_list.append(clue_indicies)

            answer_list.append(answers_vocab([answer])[0])

        answer_list = torch.tensor(answer_list).to(device)
        clue_list = torch.tensor(clue_list).to(device)

        return answer_list, clue_list

    return model, collate_batch

def load_dataset():
    # load data, split datasets, build vocabs
    #TODO: Parameterize this
    dataset = CrosswordClueAnswersDataset("cleaned_data/dupes_10_or_less_tokens.csv")
    train_size = int(0.8 * len(dataset))
    dev_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - dev_size
    g = torch.Generator().manual_seed(42) # this manual_seed is important to ensure that we consistently split the dataset
    train_dataset, test_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, dev_size], generator=g)
    print(f'{len(dataset)=}\n{len(train_dataset)=}\n{len(test_dataset)=}\n{len(dev_dataset)=}')
    return train_dataset, test_dataset, dev_dataset, dataset

def run(args):
    device = args.device
    if device == 'auto':
        #TODO: Add support for cuda
        # attempt to run on mps - will do work on the GPU for MacOS
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Running on:', device)

    # load data, split datasets, build vocabs
    train_dataset, test_dataset, dev_dataset, dataset = load_dataset()

    #model, collate_batch = build_token_model(train_dataset=train_dataset, device=device, args=args)
    model, collate_batch, vocab = build_character_model(dataset=dataset, device=device, args=args)

    # shuffle the training dataloader so we go through different batches each time
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    print(f'{len(train_dataloader)=}\n{len(dev_dataloader)=}')

    trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params_count = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_model_parameters)
    print(f'{params_count=}\n{trainable_params_count=}')

    # ignore_index is important so that we caluclate the loss ignoring padded elements
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train_dir = os.path.join('training_results', args.output_folder)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    print(f'Outputting to: {train_dir}')

    log_interval = int(len(train_dataloader) / 5)
    reporter = TrainingReporter(train_dir, log_interval)
    #training = Trainer(model=model, criterion=criterion, optimizer=optimizer, reporter=reporter, output_dir=train_dir)
    training = CharacterTrainer(model=model, criterion=criterion, optimizer=optimizer, reporter=reporter, output_dir=train_dir)
    training.start(num_epochs=args.num_epochs, train_dataloader=train_dataloader, test_dataloader=dev_dataloader, vocab=vocab)
