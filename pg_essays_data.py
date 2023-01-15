from os import listdir
from os.path import isfile, join
import torch.utils.data as data


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
        pass

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
        return 0

    def __getitem__(self, index):
        pass


if __name__ == '__main__':
    pgessays = PGEssays('pages')
