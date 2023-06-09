import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class SimpleCrosswordModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, input_size, hidden_size, output_size, device, hidden_depth=1):
        super(SimpleCrosswordModel, self).__init__()
        self.C = nn.Embedding(vocab_size, embed_dim, device=device)
        self.W1 = nn.Linear(embed_dim * input_size, hidden_size, device=device)

        self.hiddenLayers = nn.ModuleList()
        self.hiddenActivations = []
        for _ in range(hidden_depth):
            self.hiddenLayers.append(nn.Linear(hidden_size, hidden_size, device=device))

        self.W2 = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, text):
        emb = self.C(text)
        h1 = F.tanh(self.W1(emb.view(-1, self.W1.in_features)))

        for hiddenLayer in self.hiddenLayers:
            h1 = F.tanh(hiddenLayer(h1))

        return self.W2(h1)

class RecurrentCrosswordModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, output_size, device, hidden_depth=1):
        super(RecurrentCrosswordModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.embed_dim = embed_dim
        self.C = nn.Embedding(vocab_size, embed_dim, device=device)
        self.RNN = nn.RNN(embed_dim, hidden_size, hidden_depth, batch_first=True, nonlinearity='relu', device=device)
        self.W2 = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, encoded_clue):
        emb = self.C(encoded_clue) # [len(encoded_clue), embed_dim]
        #print(f'{encoded_clue.shape=}') # (batch size, 10)
        #print(f'{emb.shape=}') # (batch size, 10, 2)
        #emb_packed = emb.view(-1, self.embed_dim * 10)
        #print(f'{emb_packed.shape=}') # (batch size, 20)

        h0 = Variable(torch.zeros(self.hidden_depth, encoded_clue.size(0), self.hidden_size)).to(self.device)
        out, _ = self.RNN(emb, h0)
        out = self.W2(out[:, -1, :])
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(self.device)

class CharacterRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, device, num_layers=1, embedding_dimensions=16):
        super(CharacterRNN, self).__init__()
        self.C = nn.Embedding(vocab_size, embedding_dimensions, device=device)

        # recurrent neural network
        #self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='tanh', batch_first=True, device=device)
        self.rnn = nn.LSTM(input_size=embedding_dimensions, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, device=device)

        self.dropout = nn.Dropout(p=0.2)

        # a fully-connect layer that outputs a distribution over
        # the next token, given the RNN output
        self.decoder = nn.Linear(hidden_size, vocab_size, device=device)

    def forward(self, encoded_characters, hidden=None):
        #print(f'{encoded_characters.shape=}')
        emb = self.C(encoded_characters)
        #print(f'{emb.shape=}')
        #view = emb.view(-1, self.rnn.input_size)
        #print(f'{view.shape=}')
        #torch.nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True, enforce_sorted=False)
        #output, hidden = self.rnn(packed_one_hot_character, hidden) # get the next output and hidden state
        #output = self.decoder(torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0])          # predict distribution over next tokens
        output, hidden = self.rnn(emb, hidden) # get the next output and hidden state
        output = self.dropout(output)
        output = self.decoder(output)          # predict distribution over next tokens
        return output, hidden
