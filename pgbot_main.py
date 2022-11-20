import argparse
from bs4 import BeautifulSoup as BS
import numpy as np
from os import listdir
from os.path import isfile, join
import requests
from random import sample
import re
from statistics import mean
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


parser = argparse.ArgumentParser(
                    prog = 'pgbot',
                    description = 'Attempted bot based on Paul Graham essays.')
parser.add_argument('--fetch_essay_pages', action=argparse.BooleanOptionalAction)
parser.add_argument('--write_to_files', action=argparse.BooleanOptionalAction)
parser.add_argument('--load_essay_pages', action=argparse.BooleanOptionalAction)
parser.add_argument('--train_model', action=argparse.BooleanOptionalAction)
parser.add_argument('--load_model', action=argparse.BooleanOptionalAction)
parser.add_argument('--validate_model', action=argparse.BooleanOptionalAction)
parser.add_argument('--generate_sentence', action=argparse.BooleanOptionalAction)

BASE_URL = 'http://www.paulgraham.com/'
ARTICLES_URL = BASE_URL + 'articles.html'
PAGES = 'pages/'
PAGE_MIN_SIZE = 1000
ARGS = parser.parse_args()
MAX_TOKENS = 2000
EMBED_SIZE = 20
HIDDEN_LAYER_SIZE = 128
CONTEXT_SIZE = 3
EPOCHS = 10

MINUS_3 = '<-3>'
MINUS_2 = '<-2>'
MINUS_1 = '<-1>'
END = '<end>'

def _fetch_essay_pages():
    response = requests.get(ARTICLES_URL)
    if response.status_code == 200:
        print('Fetched the articles page.')
    else:
        print(f'Request failed with code {response.status_code}')
        return

    pgsoup = BS(response.text, 'lxml')
    print(f"Total articles fetched: {len(pgsoup.select('a')[1:])}")

    all_links = [BASE_URL + link.get('href') for link in pgsoup.find_all('a')]
    success = 0
    failure = 0
    finished_requests = 0
    all_pages = []
    total_requests = len(all_links)
    for link in all_links:
        finished_requests = finished_requests + 1
        rs = requests.get(link)
        if finished_requests % 10 == 0:
            print(f'Finished {finished_requests} out of {total_requests} requests')
        if rs.status_code == 200:
            success = success + 1
            pg_text = BS(rs.text, features="lxml").find('table').text
            pg_text = pg_text.replace("'", "").replace('\r', ' ').replace('\n', ' ')
            pg_text = re.sub('e\.g\.', 'eg', pg_text)
            pg_text = re.sub('\.', '.\n', pg_text)
            if len(pg_text) < PAGE_MIN_SIZE:
                continue
            all_pages.append(pg_text)
        else: 
            failure = failure + 1
    print(f'total_success: {success}, total_failure: {failure}')
    if ARGS.write_to_files:
        count = 1
        for p in all_pages:
            with open(PAGES + 'page-' + str(count) + '.txt', 'w') as f:
                f.write(p)
                count = count + 1
    return all_pages

def _load_essay_pages():
    all_pages = []
    for p in listdir(PAGES):
        path = join(PAGES, p)
        if isfile(path):
            with open(path, 'r') as f:
                all_pages.append(f.read())
    return all_pages

def _yield_tokens(essays):
    for e in essays:
        for s in e.split('\n'):
            yield s.split()

def _tokensize(sentence):
    tokens = [MINUS_3, MINUS_2, MINUS_1]
    tokens.extend(sentence.split())
    tokens.append('<end>')
    return tokens

def _make_ngrams(tokens):
    return [([tokens[i - j - 1] for j in range(CONTEXT_SIZE)], tokens[i]) for i in range(CONTEXT_SIZE, len(tokens))]

def _get_ngrams(essays):
    vocab = build_vocab_from_iterator(
        _yield_tokens(essays), 
        max_tokens=MAX_TOKENS, 
        specials=['<unk>', MINUS_3, MINUS_2, MINUS_1, END])
    vocab.set_default_index(0)
    sentances = [s for p in essays for s in p.split('\n') if len(s.split(' ')) > 3]
    tokenized_sentences = [_tokensize(s) for s in sentances]
    ngrams = [_make_ngrams(ts) for ts in tokenized_sentences]
    n1 = int(0.8 * len(ngrams))
    n2 = int(0.9 * len(ngrams))
    ngrams_train = ngrams[0:n1]
    ngrams_valid = ngrams[n1:n2]
    ngrams_test = ngrams[n2:]
    return (ngrams_train, ngrams_valid, ngrams_test), vocab

def _map_ngram_to_index(ng, vocab):
    X = torch.tensor([vocab(n[0]) for n in ng], dtype=torch.long)
    y = torch.tensor([vocab[n[1]] for n in ng], dtype=torch.long)
    return X, y
    
# Setup NN model
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, HIDDEN_LAYER_SIZE)
        self.linear2 = nn.Linear(HIDDEN_LAYER_SIZE, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def _train_model(ngrams, vocab, model, loss_function, optimizer):
    losses = []
    steps = 0
    for epoch in range(EPOCHS):
        for ng in ngrams:
            steps = steps + 1
            model.zero_grad()
            X, y = _map_ngram_to_index(ng, vocab)
            log_probs = model(X)
            loss = loss_function(log_probs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if steps % 1000 == 0:
                print(f'{steps} steps done.')
                print(f'avg_loss: {mean(losses[-999:]):.2f}')

def _run_validation(ngrams, vocab, model, loss_function, optimizer):
    losses = []
    steps = 0
    model.zero_grad()
    for ng in ngrams:
        X, y = _map_ngram_to_index(ng, vocab)
        loss = loss_function(model(X), y)
        losses.append(loss.item())
    return mean(losses)        


def _generate_sentence(model, vocab, max_length=50):
    sentence = []
    input = [MINUS_1, MINUS_2, MINUS_3]
    nt = ''
    while (len(sentence) < max_length) and (nt != END):    
        indexes = torch.tensor(vocab(input), dtype=torch.long).view(1, 3)
        log_probs = model(indexes)
        d = torch.distributions.categorical.Categorical(logits=log_probs)
        ind = 0
        while ind == 0:
            ind = d.sample()
        nt = vocab.lookup_token(ind)
        sentence.append(nt)
        input = [nt] + input[:-1]
    return ' '.join(sentence) + '.'




def main():
    print('Running pgbot.')
    all_pages = []
    if ARGS.fetch_essay_pages:
        all_pages = _fetch_essay_pages()
    elif ARGS.load_essay_pages:
        all_pages = _load_essay_pages()
    else:
        print('Neither fetch nor load chosen.')
    (ngrams_train, ngrams_valid, ngrams_test), vocab = _get_ngrams(all_pages)

    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBED_SIZE, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model_path = 'models/ngram-model-4'

    if ARGS.train_model:
        _train_model(ngrams_train, vocab, model, loss_function, optimizer)
        torch.save(model.state_dict(), model_path)

    if ARGS.load_model:
        model.load_state_dict(torch.load(model_path))
    
    if ARGS.validate_model:
        valid_loss = _run_validation(ngrams_valid, vocab, model, loss_function, optimizer)
        print(f'validation_loss: {valid_loss:.2f}')

    if ARGS.generate_sentence:
        for i in range(20):
            print(_generate_sentence(model, vocab))

if __name__ == "__main__":
    main()