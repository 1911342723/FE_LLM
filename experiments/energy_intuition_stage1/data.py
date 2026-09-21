"""三着色任务数据生成：保证有解、防泄漏、同构去重。"""
from __future__ import annotations
import hashlib
import numpy as np

N_NODES = 16
N_COLORS = 3


def _wl_hash(adj: np.ndarray, rounds: int = 3) -> str:
    n = adj.shape[0]
    labels = [b"1"] * n
    for _ in range(rounds):
        new = []
        for i in range(n):
            nb = sorted(labels[j] for j in range(n) if adj[i, j])
            new.append(hashlib.md5(labels[i] + b"|" + b",".join(nb)).digest())
        labels = new
    return hashlib.md5(b"".join(sorted(labels))).hexdigest()


def _exact_iso(a: np.ndarray, b: np.ndarray) -> bool:
    """16 节点精确同构判定，先用度序列剪枝，再回溯匹配。"""
    if a.sum() != b.sum():
        return False
    da, db = a.sum(1), b.sum(1)
    if sorted(da.tolist()) != sorted(db.tolist()):
        return False
    n = a.shape[0]
    order = sorted(range(n), key=lambda i: -da[i])
    mapping = [-1] * n
    used = [False] * n

    def bt(k: int) -> bool:
        if k == n:
            return True
        u = order[k]
        for v in range(n):
            if used[v] or da[u] != db[v]:
                continue
            ok = True
            for j in range(k):
                w = order[j]
                if a[u, w] != b[v, mapping[w]]:
                    ok = False
                    break
            if ok:
                mapping[u] = v
                used[v] = True
                if bt(k + 1):
                    return True
                used[v] = False
                mapping[u] = -1
        return False

    return bt(0)


def gen_graph(rng: np.random.Generator, density: float = 0.35):
    """先生成隐藏三色分组，只在不同组间连边 -> 保证有解。"""
    groups = rng.integers(0, N_COLORS, size=N_NODES)
    while len(np.unique(groups)) < N_COLORS:
        groups = rng.integers(0, N_COLORS, size=N_NODES)
    adj = np.zeros((N_NODES, N_NODES), dtype=np.int8)
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if groups[i] != groups[j] and rng.random() < density:
                adj[i, j] = adj[j, i] = 1
    # 防泄漏：随机打乱节点编号
    perm = rng.permutation(N_NODES)
    adj = adj[np.ix_(perm, perm)]
    groups = groups[perm]
    return adj, groups


def build_dataset(n: int, rng: np.random.Generator, density: float,
                  seen: dict[str, list[np.ndarray]]):
    """生成 n 张互不同构、且与 seen 中已有图不同构的图。"""
    out = []
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > n * 200:
            raise RuntimeError("图去重失败次数过多，请降低 n 或调整密度")
        adj, groups = gen_graph(rng, density)
        h = _wl_hash(adj)
        bucket = seen.setdefault(h, [])
        if any(_exact_iso(adj, prev) for prev in bucket):
            continue
        bucket.append(adj)
        out.append((adj, groups))
    return out


def make_splits(seed: int, n_train=5000, n_val=1000, n_test=1000,
                density=0.35, ood_density=0.5, n_ood=1000):
    rng = np.random.default_rng(seed)
    seen: dict[str, list[np.ndarray]] = {}
    train = build_dataset(n_train, rng, density, seen)
    val = build_dataset(n_val, rng, density, seen)
    test = build_dataset(n_test, rng, density, seen)
    ood = build_dataset(n_ood, rng, ood_density, seen)
    return {"train": train, "val": val, "test": test, "ood": ood}
