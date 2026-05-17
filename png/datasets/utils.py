# Dataset wrappers shared by multiple Lightning data modules.
from torch.utils.data import Dataset

class RepeatDataset():
    def __init__(self, dataset, times, iterations=None, batch_size=-1):
        self.dataset = dataset
        self.times = times
        self.iterations = iterations
        self.batch_size = batch_size 
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        if self.iterations is None:
            return self.times * self._ori_len
        else:
            return self.iterations * self.batch_size

class DummyDataset(Dataset):
    def __init__(self, length=1):
        super().__init__()
        self.length = length
    def __getitem__(self, idx):
        return {'dummy': 0}
    def __len__(self):
        return self.length
