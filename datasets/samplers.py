import torch
from torch.utils import data


class NShotTaskSampler(data.Sampler):
    """Sample k-way, n-shot, q-query tasks.
    First n*k samples are from support set, and the remaining q*k samples are from query set.

    Support zero-shot with n=0.

    Args:
        label: Iterable
            labels of the dataset
        k, n, q: int
        num_batches: int
            number of batches, if specified, these batches will be fixed
    """

    def __init__(self, label, k: int, n: int, q: int, num_batches=0):
        self.classes = None
        self.label_map = None
        self.k = k
        self.n = n
        self.q = q
        self.num_batches = num_batches
        self.m_ind = [[] for _ in range(max(label) + 1)]
        for i, l in enumerate(label):
            self.m_ind[l].append(i)
        self.curr_idx = 0
        if self.num_batches > 0:
            self.all_batches = []
            for _ in range(self.num_batches):
                batch = self._create_indices()
                self.all_batches.append((self.classes, self.label_map, batch))

    def __len__(self):
        return self.k * (self.n + self.q)

    def _create_indices(self):
        # sample k classes
        self.classes = torch.randperm(len(self.m_ind))[: self.k]
        self.label_map = dict([(y.item(), x) for x, y in enumerate(self.classes)])
        ind = []
        for c in self.classes:
            l = self.m_ind[c]
            # sample n+q indices
            pos = torch.randperm(len(l))[: self.n + self.q]
            ind.append(list(map(lambda x: l[x], pos)))
        mat = torch.tensor(ind)
        return mat.T.flatten()

    def __iter__(self):
        if self.num_batches > 0:
            if self.curr_idx >= self.num_batches:
                self.curr_idx = 0
            self.classes, self.label_map, batch = self.all_batches[self.curr_idx]
            self.curr_idx += 1
        else:
            batch = self._create_indices()
        return batch.__iter__()

    def get_class_labels(self, classlist):
        return list(map(lambda x: classlist[x], self.classes))

    def convert_target(self, target: torch.Tensor):
        return target.apply_(lambda x: self.label_map[x])
