import numpy as np

# ---------- Utilities ----------
def complement_code(x):
    """Complement code a vector: [x, 1-x]. Assumes x in [0,1]."""
    x = np.asarray(x, dtype=float)
    return np.concatenate([x, 1.0 - x], axis=-1)

def l1_norm(x):
    """L1 norm |x|_1."""
    return float(np.sum(x))

def fuzzy_and(a, b):
    """Fuzzy AND (componentwise min)."""
    return np.minimum(a, b)


# ---------- Fuzzy ART ----------
class FuzzyART:
    """
    Fuzzy ART with complement-coded inputs (caller supplies complement_code).

    Category choice (Eq.):
        T_j = | I ^ w_j | / (alpha + | w_j |)
      where "^" is Fuzzy AND (min), and |.| is L1 norm.

    Vigilance test (Eq.):
        | I ^ w_j | / | I | >= rho

    Learning (Eq.):
        w_j <- beta * (I ^ w_j) + (1 - beta) * w_j

    Uncommitted node:
        A node with w = ones is implicitly added when needed.
    """
    def __init__(self, input_dim_cc, alpha=0.01, beta=1.0, rho=0.8):
        self.input_dim_cc = int(input_dim_cc)  # dimension AFTER complement coding
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.rho   = float(rho)
        self.weights = []  # list[np.ndarray] — each is a prototype w_j

    # ---- internals ----
    def _ensure_uncommitted(self):
        if not self.weights:
            # standard: an uncommitted node with w=1s
            self.weights.append(np.ones(self.input_dim_cc, dtype=float))

    def _choice_values(self, I_cc):
        """Category choice values T_j for all nodes."""
        self._ensure_uncommitted()
        T = []
        for w in self.weights:
            num = l1_norm(fuzzy_and(I_cc, w))
            den = self.alpha + l1_norm(w)
            T.append(num / den)
        return np.asarray(T, dtype=float)

    def _resonance_ratio(self, I_cc, j):
        """Resonance ratio r = |I ^ w_j| / |I| (vigilance test quantity)."""
        w = self.weights[j]
        return l1_norm(fuzzy_and(I_cc, w)) / max(l1_norm(I_cc), 1e-9)

    def _learn_into(self, j, I_cc):
        """Learning rule w_j <- beta*(I^w_j) + (1-beta)*w_j."""
        self.weights[j] = self.beta * fuzzy_and(I_cc, self.weights[j]) + (1.0 - self.beta) * self.weights[j]

    def _add_and_learn(self, I_cc):
        """Create a new (uncommitted) node and immediately learn into it (standard Fuzzy ART)."""
        self.weights.append(np.ones(self.input_dim_cc, dtype=float))
        j = len(self.weights) - 1
        self._learn_into(j, I_cc)
        return j

    # ---- training-time search (with learning & node creation) ----
    def process_train(self, I_cc, rho=None, forbid=set()):
        """
        Category choice → ordered search → vigilance → learn.
        If none passes vigilance, create a new node.
        Returns: (winner_index, compressed_code_Z, resonance_ratio)
        """
        if rho is None:
            rho = self.rho
        self._ensure_uncommitted()
        forbid = set(forbid)

        # Visit categories by descending T_j (largest choice first — standard).
        order = np.argsort(-self._choice_values(I_cc))

        for j in order:
            if j in forbid:   # ARTMAP-style reset forbids previously tried j’s
                continue
            rr = self._resonance_ratio(I_cc, j)
            if rr >= rho:
                # Compressed code Z (recognition code) using old weight (beta=1 => Z=I^w_old)
                w_old = self.weights[j].copy()
                Z = (1.0 - self.beta) * w_old + self.beta * fuzzy_and(I_cc, w_old)
                # Learn
                self._learn_into(j, I_cc)
                return j, Z, rr

        # No category met vigilance: allocate a new category and learn
        j_new = self._add_and_learn(I_cc)
        w_old = np.ones_like(self.weights[j_new])
        Z = (1.0 - self.beta) * w_old + self.beta * fuzzy_and(I_cc, w_old)
        rr = self._resonance_ratio(I_cc, j_new)
        return j_new, Z, rr

    # ---- test-time search (no learning, no node creation) ----
    def process_infer(self, I_cc, rho=None, forbid=set()):
        """
        Category choice → ordered search → vigilance. NO learning. NO node creation.
        Returns: (winner_index or None, Z (using current w), resonance_ratio or -inf)
        """
        if rho is None:
            rho = self.rho
        self._ensure_uncommitted()
        forbid = set(forbid)

        order = np.argsort(-self._choice_values(I_cc))
        for j in order:
            if j in forbid:
                continue
            rr = self._resonance_ratio(I_cc, j)
            if rr >= rho:
                w = self.weights[j]
                Z = (1.0 - self.beta) * w + self.beta * fuzzy_and(I_cc, w)
                return j, Z, rr

        return None, np.zeros(self.input_dim_cc, dtype=float), float("-inf")
