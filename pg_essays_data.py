from bs4 import BeautifulSoup as BS
from os import listdir
from os.path import isfile, join
import re
import requests
import torch.utils.data as data

BASE_URL = 'http://www.paulgraham.com/'
ARTICLES_URL = BASE_URL + 'articles.html'
PAGE_MIN_SIZE = 1000


class PGEssays(data.Dataset):

    def __load_from_path(self):
        all_pages = []
        for p in listdir(self.pages_path):
            path = join(self.pages_path, p)
        if isfile(path):
            with open(path, 'r') as f:
                all_pages.append(f.read())
        return all_pages

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

    def __init__(self, path, load_from_path=True, fetch_from_web=False):
        super().__init__()
        if (not load_from_path) and (not fetch_from_web):
            raise Exception(
                'load_from_path and fetch_from_web cannot both be false.')
        self.essays = []
        self.pages_path = path
        if load_from_path:
            self.essays = self.__load_from_path()
        else:
            self.essays = self.__fetch_from_web()

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, index):
        return self.essays[index]


if __name__ == '__main__':
    pgessays = PGEssays('pages')
