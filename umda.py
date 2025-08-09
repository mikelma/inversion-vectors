import numpy as np
from enum import Enum

class SampleOrder(Enum):
    Left = 1,
    Right = 2,
    Random = 3,

class Umda():
    def __init__(self):
        self.distrib = None
        self.n = None

    def learn(self, pop):
        pop = np.array(pop)
        n = pop.shape[1]
        distrib = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distrib[i, j] = sum(pop[:, i] == j)
        # normalize
        distrib /= distrib.sum(axis=1)
        self.distrib = distrib
        self.n = n

    def sample(self):
        assert self.distrib is not None, "learn method must be called before sampling"
        return np.array([np.random.choice(self.n, p=p) for p in self.distrib])

    def sample_permu(self, sample_order=SampleOrder.Right):
        assert self.distrib is not None, "learn method must be called before sampling"

        if sample_order == SampleOrder.Right:
            order = np.arange(self.n, dtype=np.int64)
        elif sample_order == SampleOrder.Left:
            order = np.arange(self.n, dtype=np.int64)[::-1]
        else:
            order = np.random.permutation(self.n)
        p = np.empty(self.n, dtype=np.int64)
        e = np.arange(self.n, dtype=np.int64)
        pending = np.full(self.n, True)
        for i in order:
            d = self.distrib[i]
            probs = d[pending] + 1e-5
            probs = np.nan_to_num(probs / probs.sum())
            v = np.random.choice(e[pending], p=probs)
            p[i] = v
            pending[v] = False
        return p
