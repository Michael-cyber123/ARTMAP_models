import numpy as np
from functools import partial
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler

l1_norm = partial(np.linalg.norm, ord=1, axis=-1)


# ========================
# Fuzzy ART
# ========================
class FuzzyART:
    def __init__(self, alpha=1.0, rho=0.0, gamma=0.01, complement_coding=True):
        self.alpha = float(alpha)
        self.beta = 1.0 - float(alpha)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.complement_coding = bool(complement_coding)
        self.w = None

    def _init_weights(self, x):
        M = x.shape[0]
        self.w = np.ones((1, M), dtype=np.float32)

    def _complement_code(self, x):
        if self.complement_coding:
            return np.hstack((x, 1.0 - x))
        return x

    def _add_category(self, x):
        if self.w is None:
            self.w = np.atleast_2d(x).astype(np.float32)
        else:
            self.w = np.vstack((self.w, x)).astype(np.float32)

    def _match_category(self, x):
        x = np.atleast_2d(x).astype(np.float32)
        if self.w is None:
            return -1

        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))

        denom = l1_norm(x) + 1e-12
        threshold = (fuzzy_norm / denom) >= self.rho

        if np.all(threshold == False):
            return -1
        return int(np.argmax(scores * threshold.astype(int)))

    def train(self, x, epochs=1):
        samples = self._complement_code(np.atleast_2d(x).astype(np.float32))
        if self.w is None:
            self._init_weights(samples[0])

        for _ in range(int(epochs)):
            for sample in np.random.permutation(samples):
                category = self._match_category(sample)
                if category == -1:
                    self._add_category(sample)
                else:
                    w = self.w[category]
                    self.w[category] = (self.alpha * np.minimum(sample, w) +
                                        self.beta * w)
                    self.w = np.clip(self.w, 0.0, None)
        return self

    def predict(self, x):
        samples = self._complement_code(np.atleast_2d(x).astype(np.float32))
        categories = np.zeros(len(samples), dtype=int)
        for i, sample in enumerate(samples):
            categories[i] = self._match_category(sample)
        return categories


# ========================
# Fuzzy ARTMAP
# ========================
class FuzzyARTMAP(FuzzyART):
    def __init__(self, alpha=1.0, rho=0.0, gamma=0.01, epsilon=1e-4, complement_coding=True):
        self.alpha = float(alpha)
        self.beta = 1.0 - float(alpha)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.epsilon = float(epsilon)
        self.complement_coding = bool(complement_coding)
        self.w = None
        self.out_w = None
        self.n_classes = 0

    def _init_weights(self, M):
        self.w = np.ones((1, M), dtype=np.float32)
        self.out_w = np.zeros((1, self.n_classes), dtype=np.float32)

    def _complement_code(self, x):
        if self.complement_coding:
            return np.hstack((x, 1.0 - x))
        return x

    def _add_category(self, x, y):
        if self.w is None:
            self._init_weights(x.shape[0])
            self.w[0] = x
        else:
            self.w = np.vstack((self.w, x)).astype(np.float32)
            self.out_w = np.vstack((self.out_w, np.zeros((1, self.n_classes), dtype=np.float32)))

        self.out_w[-1, int(y)] = 1.0

    def _match_category(self, x, y=None, return_scores=False):
        x = np.atleast_2d(x).astype(np.float32)
        if self.w is None:
            return (-1, None) if return_scores else -1

        _rho = self.rho

        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))

        denom = l1_norm(x) + 1e-12
        norms = fuzzy_norm / denom
        threshold = norms >= _rho

        while not np.all(threshold == False):
            j = int(np.argmax(scores * threshold.astype(int)))
            if y is None or self.out_w[j, int(y)] == 1.0:
                return (j, scores) if return_scores else j
            else:
                _rho = float(norms[j] + self.epsilon)
                norms[j] = 0.0
                threshold = norms >= _rho

        return (-1, None) if return_scores else -1

    def train(self, x, y, epochs=1):
        x = np.atleast_2d(x).astype(np.float32)
        y = np.asarray(y, dtype=int)

        # IMPORTANT: set n_classes before init_weights
        self.n_classes = int(len(set(y.tolist())))
        samples = self._complement_code(x)

        if self.w is None:
            self._init_weights(samples.shape[1])

        idx = np.arange(len(samples), dtype=np.uint32)

        for _ in range(int(epochs)):
            idx = np.random.permutation(idx)
            for sample, label in zip(samples[idx], y[idx]):
                category = self._match_category(sample, label)
                if category == -1:
                    self._add_category(sample, label)
                else:
                    w = self.w[category]
                    self.w[category] = (self.alpha * np.minimum(sample, w) +
                                        self.beta * w)
                    self.w = np.clip(self.w, 0.0, None)
        return self

    def predict(self, x):
        samples = self._complement_code(np.atleast_2d(x).astype(np.float32))
        labels = np.zeros(len(samples), dtype=int)
        match_scores = np.zeros(len(samples), dtype=float)

        for i, sample in enumerate(samples):
            category, scores = self._match_category(sample, return_scores=True)
            if category != -1:
                labels[i] = int(np.argmax(self.out_w[category]))
                match_scores[i] = float(scores[category])
            else:
                labels[i] = -1
                match_scores[i] = 0.0

        return labels, match_scores


# ============================================================
# Convenience wrappers: MinMax scaling + fallback label
# ============================================================
def fit_fam(X_train, y_train, alpha=1.0, rho=0.0, gamma=0.01, epsilon=1e-4, epochs=1, complement_coding=True):
    """
    Fit a single FAM model on MinMax-scaled data in [0,1].
    Returns: fam, scaler, fallback_label
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)

    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)

    fam = FuzzyARTMAP(alpha=alpha, rho=rho, gamma=gamma, epsilon=epsilon, complement_coding=complement_coding)
    fam.train(Xtr, y_train, epochs=epochs)

    fallback_label = int(mode(y_train, keepdims=False).mode)
    return fam, scaler, fallback_label


def predict_fam(fam, scaler, X_test, fallback_label=0):
    """
    Predict with a single FAM model. Returns y_pred, y_score (match score).
    """
    X_test = np.asarray(X_test, dtype=np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)

    y_pred, y_score = fam.predict(Xte)
    y_pred = np.where(y_pred == -1, fallback_label, y_pred).astype(int)
    return y_pred, np.asarray(y_score, dtype=float)
