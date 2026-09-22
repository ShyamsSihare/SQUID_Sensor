import numpy as np

RY = lambda th: np.array([[np.cos(th / 2), -np.sin(th / 2)],
                           [np.sin(th / 2), np.cos(th / 2)]], dtype=complex)
RZ = lambda th: np.array([[np.exp(-1j * th / 2), 0],
                           [0, np.exp(1j * th / 2)]], dtype=complex)


def apply_1q_batch(state, gate, qubit):
    ax = qubit + 1
    state = np.moveaxis(state, ax, 1)
    state = np.einsum('ij,bj...->bi...', gate, state)
    state = np.moveaxis(state, 1, ax)
    return state


def apply_cz_batch(state, q1, q2):
    idx = [slice(None)] * state.ndim
    idx[q1 + 1] = 1
    idx[q2 + 1] = 1
    state = state.copy()
    state[tuple(idx)] *= -1
    return state


def brickwork_pairs(n):
    pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
    pairs += [(i, i + 1) for i in range(1, n - 1, 2)]
    return pairs


def n_params(n, L):
    return 2 * n * L  # RY + RZ per qubit per layer


def ansatz_batch(state, theta, n, L):
    idx = 0
    pairs = brickwork_pairs(n)
    for layer in range(L):
        for q in range(n):
            state = apply_1q_batch(state, RY(theta[idx]), q); idx += 1
            state = apply_1q_batch(state, RZ(theta[idx]), q); idx += 1
        for (q1, q2) in pairs:
            state = apply_cz_batch(state, q1, q2)
    return state


REF_TOTAL_POWER = 4.0


def encode_batch(X, n, ref_total_power=REF_TOTAL_POWER):
    dim = 2 ** n
    main_bins = dim - 1
    B, F = X.shape
    if F >= main_bins:
        Xm = X[:, :main_bins]
    else:
        Xm = np.concatenate([X, np.zeros((B, main_bins - F))], axis=1)
    amp_main = np.sqrt(np.clip(Xm, 0, None)) / np.sqrt(ref_total_power)
    norm_sq = (amp_main ** 2).sum(axis=1)
    over = norm_sq > 0.999
    scale = np.ones_like(norm_sq)
    scale[over] = np.sqrt(0.999 / norm_sq[over])
    amp_main = amp_main * scale[:, None]
    ancilla = np.sqrt(np.clip(1 - (amp_main ** 2).sum(axis=1), 0, None))
    full = np.concatenate([amp_main, ancilla[:, None]], axis=1)
    return full.reshape((-1,) + (2,) * n).astype(complex)


def overlap_to_zero_batch(state):
    flat = state.reshape(state.shape[0], -1)
    return np.abs(flat[:, 0]) ** 2


def overlap_to_target_batch(state, target_flat):
    flat = state.reshape(state.shape[0], -1)
    ov = flat @ np.conj(target_flat)
    return np.abs(ov) ** 2


def forward_scores(theta, X, n, L, target_flat=None):
    s = encode_batch(X, n)
    s = ansatz_batch(s, theta, n, L)
    if target_flat is None:
        return 1.0 - overlap_to_zero_batch(s)
    return 1.0 - overlap_to_target_batch(s, target_flat)


def batch_loss(theta, X, n, L, target_flat=None):
    return float(np.mean(forward_scores(theta, X, n, L, target_flat)))


def parameter_shift_grad(theta, X, n, L, target_flat=None):
    grad = np.zeros_like(theta)
    for k in range(len(theta)):
        tp = theta.copy(); tp[k] += np.pi / 2
        tm = theta.copy(); tm[k] -= np.pi / 2
        grad[k] = 0.5 * (batch_loss(tp, X, n, L, target_flat) - batch_loss(tm, X, n, L, target_flat))
    return grad


def train_vqc(X_train, n, L, lr, n_iters, batch_size, seed, verbose=False):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-0.3, 0.3, size=n_params(n, L))
    ref_idx = rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    ref_states = encode_batch(X_train[ref_idx], n).reshape(len(ref_idx), -1)
    target_flat = ref_states.mean(axis=0)
    target_flat = target_flat / np.linalg.norm(target_flat)

    N = len(X_train)
    losses = []
    for it in range(n_iters):
        idx = rng.choice(N, size=min(batch_size, N), replace=False)
        batch = X_train[idx]
        grad = parameter_shift_grad(theta, batch, n, L, target_flat)
        theta -= lr * grad
        if verbose and (it % max(1, n_iters // 5) == 0 or it == n_iters - 1):
            print(f"  iter {it:4d}  loss={batch_loss(theta, batch, n, L, target_flat):.4f}")
    losses.append((n_iters, batch_loss(theta, X_train[:min(200, N)], n, L, target_flat)))
    return theta, losses, target_flat
