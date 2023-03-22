from torch import nn

class SimpleCrosswordModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, input_size, hidden_size, output_size, device):
        super(SimpleCrosswordModel, self).__init__()
        self.C = nn.Embedding(vocab_size, embed_dim, device=device)
        self.W1 = nn.Linear(embed_dim * input_size, hidden_size, device=device)
        self.M = nn.Tanh()
        self.W2 = nn.Linear(hidden_size, output_size, device=device)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.C.weight.data.uniform_(-initrange, initrange)
        self.W1.weight.data.uniform_(-initrange, initrange)
        self.W1.bias.data.zero_()
        self.W2.weight.data.uniform_(-initrange, initrange)
        self.W2.bias.data.zero_()

    def forward(self, text):
        emb = self.C(text)
        h = self.M(self.W1(emb.view(-1, self.W1.in_features)))
        return self.W2(h)