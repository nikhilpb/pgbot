from bs4 import BeautifulSoup as BS
import numpy as np
import requests
# import spacy
import random
import re
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


BASE_URL = 'http://www.paulgraham.com/'
ARTICLES_URL = BASE_URL + 'articles.html'
PAGES = 'pages/'
PAGE_MIN_SIZE = 1000

def fetch_essay_pages(write_to_files = False):
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
    print(f'Approximately {float(sum([len(t) for t in all_pages])) / (1024 * 1024):.2f} MB of data fetched')
    if write_to_files:
        count = 1
        for p in all_pages:
            with open(PAGES + 'page-' + str(count) + '.txt', 'w') as f:
                f.write(p)
                count = count + 1
    return all_pages


if __name__ == "__main__":
    print('Running pgbot.')
    all_pages = fetch_essay_pages(write_to_files=True)
    print('Done.')