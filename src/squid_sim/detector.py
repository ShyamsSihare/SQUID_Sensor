import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

import physics as ph


REF_TOTAL_POWER = 4.0  # calibration constant, see note below


def encode_amplitude(S, n_dims=256, ref_total_power=REF_TOTAL_POWER):
    S = np.atleast_2d(S)
    main_bins = n_dims - 1
    Smain = S[:, :main_bins]
    amp_main = np.sqrt(np.clip(Smain, 0, None)) / np.sqrt(ref_total_power)
    norm_sq = (amp_main ** 2).sum(axis=1)
    over = norm_sq > 0.999
    scale = np.ones_like(norm_sq)
    scale[over] = np.sqrt(0.999 / norm_sq[over])
    amp_main = amp_main * scale[:, None]
    ancilla = np.sqrt(np.clip(1 - (amp_main ** 2).sum(axis=1), 0, None))
    return np.concatenate([amp_main, ancilla[:, None]], axis=1)


class RelativeEntropyDetector:
    def __init__(self, reg_lambda_min=1e-9):
        self.reg_lambda_min = reg_lambda_min
        self.eigvals = None
        self.eigvecs = None

    def fit(self, S_train):
        psi = encode_amplitude(S_train)  # (M, 256)
        M = psi.shape[0]
        rho = (psi.T @ psi) / M  # (256,256) real symmetric density matrix
        w, v = np.linalg.eigh(rho)
        w = np.clip(w, 0, None)
        w = w / w.sum()  # renormalize trace=1 after clipping
        self.eigvals = w
        self.eigvecs = v
        return self

    def score(self, S_test):
        psi = encode_amplitude(S_test)  # (N, 256)
        overlaps = (psi @ self.eigvecs) ** 2  # (N, 256), |<v_k|psi_t>|^2
        lam = np.clip(self.eigvals, self.reg_lambda_min, None)
        neg_ln_lam = -np.log(lam)
        return overlaps @ neg_ln_lam


def spectral_features(S):
    S = np.atleast_2d(S)
    freq = ph.FREQ_BINS
    logf = np.log10(freq)
    total = S.sum(axis=1)
    low_band = S[:, freq < 1.0].sum(axis=1)
    high_band = S[:, freq >= 1.0].sum(axis=1)
    centroid = (S * logf).sum(axis=1) / np.clip(S.sum(axis=1), 1e-30, None)
    peak_ratio = S.max(axis=1) / np.clip(S.mean(axis=1), 1e-30, None)
    # crude log-log slope via least squares per row
    logS = np.log10(np.clip(S, 1e-30, None))
    X = np.vstack([logf, np.ones_like(logf)]).T
    slopes = np.array([np.linalg.lstsq(X, row, rcond=None)[0][0] for row in logS])
    feats = np.vstack([total, low_band, high_band, centroid, peak_ratio, slopes]).T
    return feats


class ClassicalOCSVM:
    def __init__(self, nu=0.02, gamma="scale"):
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

    def fit(self, S_train):
        X = spectral_features(S_train)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def score(self, S_test):
        X = spectral_features(S_test)
        Xs = self.scaler.transform(X)
        # decision_function: positive = inlier: negate so higher = more anomalous
        return -self.model.decision_function(Xs)


def roc_auc(scores, labels):
    order = np.argsort(-scores)
    scores_s = scores[order]
    labels_s = labels[order]
    P = labels.sum()
    N = len(labels) - P
    tps = np.cumsum(labels_s == 1)
    fps = np.cumsum(labels_s == 0)
    tpr = tps / P
    fpr = fps / N
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, scores_s, auc
