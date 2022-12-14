import torch.nn as nn
import torch.nn.functional as F


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_layer_size, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(
            context_size * embedding_dim, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
