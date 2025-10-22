# dataset_and_loader.py（在前面版本基础上加 normalize_state）
import glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class NpyTransitionDataset(Dataset):
    def __init__(self, pattern: str, normalize_state: bool = True):
        self.items = []
        for f in sorted(glob.glob(pattern)):
            obj = np.load(f, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.shape == ():
                obj = obj.item()
            s = np.asarray(obj["s"], dtype=np.float32)
            a = np.asarray(obj["a"], dtype=np.float32)
            r = np.asarray(obj["r"], dtype=np.float32).reshape(-1, 1)
            q = np.asarray(obj["q"], dtype=np.float32).reshape(-1, 1)
            n = min(len(s), len(a), len(r), len(q))
            self.items.append((
                torch.from_numpy(s[:n]),
                torch.from_numpy(a[:n]),
                torch.from_numpy(r[:n]),
                torch.from_numpy(q[:n]),
            ))

        self.s = torch.cat([t[0] for t in self.items], dim=0)  # (N, 513)
        self.a = torch.cat([t[1] for t in self.items], dim=0)  # (N, 54)
        self.r = torch.cat([t[2] for t in self.items], dim=0)
        self.q = torch.cat([t[3] for t in self.items], dim=0)

        self.s_dim, self.a_dim = self.s.shape[1], self.a.shape[1]

        # 只标准化 s
        if normalize_state:
            self.s_mean = self.s.mean(dim=0, keepdim=True)
            self.s_std  = self.s.std(dim=0, keepdim=True).clamp_min(1e-6)
        else:
            self.s_mean = torch.zeros(1, self.s_dim)
            self.s_std  = torch.ones(1, self.s_dim)

    def __len__(self): return self.s.size(0)

    def __getitem__(self, idx):
        s = (self.s[idx] - self.s_mean.squeeze(0)) / self.s_std.squeeze(0)
        return {"s": s, "a": self.a[idx], "r": self.r[idx], "q": self.q[idx]}

def dataloader(pattern="./*.npy", batch_size=1024, shuffle=True, num_workers=0):
    ds = NpyTransitionDataset(pattern, normalize_state=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return ds, dl
