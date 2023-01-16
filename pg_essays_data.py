from bs4 import BeautifulSoup as BS
from os import listdir
from os.path import isfile, join
import re
import requests
import torch.utils.data as data
from torchtext.vocab import build_vocab_from_iterator

BASE_URL = 'http://www.paulgraham.com/'
ARTICLES_URL = BASE_URL + 'articles.html'
MAX_TOKENS = 2000
PAGE_MIN_SIZE = 1000
PAGES = 'pages/'

START = '<start>'
END = '<end>'
UNK = '<unk>'


class PGEssaysNgrams(data.Dataset):

    def __load_from_path(self):
        essays = []
        for p in listdir(self.pages_path):
            path = join(self.pages_path, p)
            if isfile(path):
                with open(path, 'r') as f:
                    essays.append(f.read())
        return essays

    def __fetch_from_web(self):
        response = requests.get(ARTICLES_URL)
        if response.status_code == 200:
            print('Fetched the articles page.')
        else:
            print(f'Request failed with code {response.status_code}')
            return []
        pgsoup = BS(response.text, 'lxml')
        print(f"Total articles fetched: {len(pgsoup.select('a')[1:])}")

        all_links = [BASE_URL + link.get('href')
                     for link in pgsoup.find_all('a')]
        success = 0
        failure = 0
        finished_requests = 0
        all_pages = []
        total_requests = len(all_links)
        for link in all_links:
            finished_requests = finished_requests + 1
            rs = requests.get(link)
            if finished_requests % 10 == 0:
                print(
                    f'Finished {finished_requests} out of {total_requests} requests')
            if rs.status_code == 200:
                success = success + 1
                pg_text = BS(rs.text, features="lxml").find('table').text
                pg_text = pg_text.replace("'", "").replace(
                    '\r', ' ').replace('\n', ' ')
                pg_text = re.sub('e\.g\.', 'eg', pg_text)
                pg_text = re.sub('\.', '.\n', pg_text)
                if len(pg_text) < PAGE_MIN_SIZE:
                    continue
                all_pages.append(pg_text)
            else:
                failure = failure + 1
        print(f'total_success: {success}, total_failure: {failure}')
        count = 1
        for p in all_pages:
            with open(PAGES + 'page-' + str(count) + '.txt', 'w') as f:
                f.write(p)
                count = count + 1
        return all_pages

    def __yield_tokens(self, essays):
        for e in essays:
            for s in e.split('\n'):
                yield s.split()

    def __tokensize(self, sentence):
        tokens = [START, START, START]
        tokens.extend(sentence.split())
        tokens.append(END)
        return tokens

    def __make_ngrams(self, tokens):
        return [([tokens[i - j - 1] for j in range(self.context_size)], tokens[i]) for i in range(self.context_size, len(tokens))]

    def __init__(self, path, load_from_path=True, fetch_from_web=False, context_size=3):
        super().__init__()
        if (not load_from_path) and (not fetch_from_web):
            raise Exception(
                'load_from_path and fetch_from_web cannot both be false.')
        self.essays = []
        self.pages_path = path
        self.context_size = context_size
        if load_from_path:
            self.essays = self.__load_from_path()
        else:
            self.essays = self.__fetch_from_web()
        self.vocab = build_vocab_from_iterator(
            self.__yield_tokens(self.essays),
            max_tokens=MAX_TOKENS,
            specials=[UNK, START, END])
        self.vocab.set_default_index(0)
        sentances = [s for p in self.essays for s in p.split(
            '\n') if len(s.split(' ')) > self.context_size]
        tokenized_sentences = [self.__tokensize(s) for s in sentances]
        self.ngrams = [self.__make_ngrams(ts) for ts in tokenized_sentences]
        print(f'Essays size: {len(self.essays)}')
        print(f'Ngram size: {len(self.ngrams)}')

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, index):
        return self.ngrams[index]


if __name__ == '__main__':
    pgessays = PGEssaysNgrams('pages')
