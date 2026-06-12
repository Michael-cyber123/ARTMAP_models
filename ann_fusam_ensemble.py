import os
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle as sk_shuffle
from scipy.stats import mode
from fusion_artmap import FusionARTMAP
from fuzzy_art import complement_code

# =========================================================
# CONFIG
# =========================================================
EMBED_DIM = 4
EPOCHS_FOLD0 = 20
EPOCHS_ONLINE = 20
BATCH_SIZE = 32
ANN_LR = 1e-3

# Spambase channel partition
SLICE_WORD = slice(0, 48)
SLICE_CHAR = slice(48, 54)
SLICE_CAPS = slice(54, 57)

# Fusion ARTMAP parameters
FAM_ALPHA = 0.01
FAM_BETA = 1.0
FAM_RHO_C = 0.0
FAM_RHO_A = 0.0
FAM_RHO_B = 1.0
FAM_RHO_AB = 1.0
FAM_EPS = 1e-4
FAM_MAX_PMT_ITERS = 10
FAM_ARTA_COMPLEMENT = False

# Scaling mode
USE_GLOBAL_MINMAX = True

# Shuffle protocol
USE_TWO_STAGE_SHUFFLE = True


# =========================================================
# UTILS
# =========================================================
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_xy(csv_path: str, label_last_col: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    if label_last_col:
        y = df.iloc[:, -1].astype(int).values
        X = df.iloc[:, :-1].values.astype(np.float32)
    else:
        raise ValueError("Only label_last_col=True is currently supported.")

    return X, y


def build_ann(input_dim: int, embed_dim: int = EMBED_DIM, lr: float = ANN_LR, name: str = "c1"):
    x = L.Input(shape=(input_dim,), name=f"{name}_in")
    h = L.Dense(64, activation="relu", name=f"{name}_h1")(x)
    h = L.Dense(32, activation="relu", name=f"{name}_h2")(h)
    z = L.Dense(embed_dim, activation="relu", name=f"{name}_z")(h)
    yhat = L.Dense(1, activation="sigmoid", name=f"{name}_y")(z)

    model = keras.Model(x, yhat, name=f"ann_{name}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    embedder = keras.Model(x, z, name=f"embedder_{name}")
    return model, embedder


def split_channels(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return X[:, SLICE_WORD], X[:, SLICE_CHAR], X[:, SLICE_CAPS]


# =========================================================
# FUSION WRAPPER
# =========================================================
class FusionAdapter:
    """
    Wrapper for feature scaling, clipping, and Fusion ARTMAP inference.
    """

    def __init__(self, fam: FusionARTMAP, mode: str, scalers):
        self.fam = fam
        self.mode = mode
        self.scalers = scalers

    def _prep_batch(self, Z_list: List[np.ndarray]) -> List[np.ndarray]:
        if self.mode == "global":
            mins, maxs, sl1, sl2, sl3 = self.scalers
            out = []
            for Z, sl in zip(Z_list, [sl1, sl2, sl3]):
                Zt = (Z - mins[sl]) / (maxs[sl] - mins[sl] + 1e-12)
                out.append(np.clip(Zt, 0.0, 1.0))
            return out

        m1, m2, m3 = self.scalers
        return [
            np.clip(m1.transform(Z_list[0]), 0.0, 1.0),
            np.clip(m2.transform(Z_list[1]), 0.0, 1.0),
            np.clip(m3.transform(Z_list[2]), 0.0, 1.0),
        ]

    def partial_fit(self, Z1: np.ndarray, Z2: np.ndarray, Z3: np.ndarray, y: np.ndarray):
        Z1, Z2, Z3 = self._prep_batch([Z1, Z2, Z3])
        for i in range(len(y)):
            self.fam.train_one([Z1[i], Z2[i], Z3[i]], int(y[i]), verbose=False)
        return self

    def _predict_one_with_score(self, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray):
        fam = self.fam
        compressed = []

        for art, zk in zip(fam.art_channels, [z1, z2, z3]):
            I_cc = complement_code(np.clip(zk, 0.0, 1.0))
            j, Z, _ = art.process_infer(I_cc, rho=fam.rho_c, forbid=set())

            if j is None:
                T = art._choice_values(I_cc)
                j = int(np.argmax(T))
                w = art.weights[j]
                Z = (1.0 - art.beta) * w + fam.beta * np.minimum(I_cc, w)

            compressed.append(Z)

        A = np.concatenate(compressed, axis=0)
        A_in = complement_code(A) if fam.art_a_complement else A

        j_a, _, _ = fam.art_a.process_infer(A_in, rho=fam.rho_a, forbid=set())
        T_all = fam.art_a._choice_values(A_in)

        if j_a is None:
            j_a = int(np.argmax(T_all))

        if int(j_a) in fam.mapW:
            W = fam.mapW[int(j_a)]
            y_pred = int(np.argmax(W))
        else:
            order = np.argsort(-T_all)
            y_pred = 0
            for j in order:
                if int(j) in fam.mapW and fam.art_a._resonance_ratio(A_in, int(j)) >= fam.rho_a:
                    y_pred = int(np.argmax(fam.mapW[int(j)]))
                    j_a = int(j)
                    break

        pos_score = float(T_all[int(j_a)])
        return y_pred, pos_score

    def predict_with_scores(self, Z1: np.ndarray, Z2: np.ndarray, Z3: np.ndarray):
        Z1, Z2, Z3 = self._prep_batch([Z1, Z2, Z3])

        n = Z1.shape[0]
        y_pred = np.zeros(n, dtype=int)
        pos_score = np.zeros(n, dtype=float)

        for i in range(n):
            y_pred[i], pos_score[i] = self._predict_one_with_score(Z1[i], Z2[i], Z3[i])

        return y_pred, pos_score


# =========================================================
# INITIALIZATION
# =========================================================
def initialize_ann_fusion_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
):
    
    set_all_seeds(seed)

    if USE_TWO_STAGE_SHUFFLE:
        X_base, y_base = sk_shuffle(X, y, random_state=42)
        X_shuf, y_shuf = sk_shuffle(X_base, y_base, random_state=seed)
    else:
        X_shuf, y_shuf = sk_shuffle(X, y, random_state=seed)

    X_folds = np.array_split(X_shuf, 10)
    y_folds = np.array_split(y_shuf, 10)

    X0 = X_folds[0]
    y0 = y_folds[0]

    scaler0 = StandardScaler().fit(X0)
    X0s = scaler0.transform(X0).astype(np.float32)

    X0_c1, X0_c2, X0_c3 = split_channels(X0s)

    ann1, emb1 = build_ann(X0_c1.shape[1], name="c1")
    ann2, emb2 = build_ann(X0_c2.shape[1], name="c2")
    ann3, emb3 = build_ann(X0_c3.shape[1], name="c3")

    ann1.fit(X0_c1, y0, epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)
    ann2.fit(X0_c2, y0, epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)
    ann3.fit(X0_c3, y0, epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)

    Z1_0 = emb1.predict(X0_c1, verbose=0)
    Z2_0 = emb2.predict(X0_c2, verbose=0)
    Z3_0 = emb3.predict(X0_c3, verbose=0)

    if USE_GLOBAL_MINMAX:
        Z0_concat = np.concatenate([Z1_0, Z2_0, Z3_0], axis=1)
        mins = Z0_concat.min(axis=0)
        maxs = Z0_concat.max(axis=0)

        s1, s2, s3 = Z1_0.shape[1], Z2_0.shape[1], Z3_0.shape[1]
        idx1 = slice(0, s1)
        idx2 = slice(s1, s1 + s2)
        idx3 = slice(s1 + s2, s1 + s2 + s3)

        def apply_global(z, sl):
            return np.clip((z - mins[sl]) / (maxs[sl] - mins[sl] + 1e-12), 0.0, 1.0)

        Z1_0 = apply_global(Z1_0, idx1)
        Z2_0 = apply_global(Z2_0, idx2)
        Z3_0 = apply_global(Z3_0, idx3)

        scaler_pack = (mins, maxs, idx1, idx2, idx3)
        adapter_mode = "global"
    else:
        mms1 = MinMaxScaler().fit(Z1_0)
        mms2 = MinMaxScaler().fit(Z2_0)
        mms3 = MinMaxScaler().fit(Z3_0)

        Z1_0 = np.clip(mms1.transform(Z1_0), 0.0, 1.0)
        Z2_0 = np.clip(mms2.transform(Z2_0), 0.0, 1.0)
        Z3_0 = np.clip(mms3.transform(Z3_0), 0.0, 1.0)

        scaler_pack = (mms1, mms2, mms3)
        adapter_mode = "per"

    fam_core = FusionARTMAP(
        channel_dims=[EMBED_DIM, EMBED_DIM, EMBED_DIM],
        n_classes=2,
        alpha=FAM_ALPHA,
        beta=FAM_BETA,
        rho_c=FAM_RHO_C,
        rho_a=FAM_RHO_A,
        rho_b=FAM_RHO_B,
        rho_ab=FAM_RHO_AB,
        eps=FAM_EPS,
        max_pmt_iters=FAM_MAX_PMT_ITERS,
        art_a_complement=FAM_ARTA_COMPLEMENT,
        reset_vigilance_each_sample=True,
    )

    adapter = FusionAdapter(fam_core, adapter_mode, scaler_pack)
    adapter.partial_fit(Z1_0, Z2_0, Z3_0, y0)

    return adapter, [ann1, ann2, ann3], [emb1, emb2, emb3], scaler0, X_folds, y_folds


# =========================================================
# PREDICTION
# =========================================================
def predict_ann_fusion(
    adapter: FusionAdapter,
    embedders,
    scaler0: StandardScaler,
    X: np.ndarray,
):
    Xs = scaler0.transform(X).astype(np.float32)
    X_c1, X_c2, X_c3 = split_channels(Xs)

    Z1 = embedders[0].predict(X_c1, verbose=0)
    Z2 = embedders[1].predict(X_c2, verbose=0)
    Z3 = embedders[2].predict(X_c3, verbose=0)

    return adapter.predict_with_scores(Z1, Z2, Z3)


# =========================================================
# ONLINE EVALUATION
# =========================================================
def run_online_ann_fusion(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,)   
    Tuple[float, List[float], pd.DataFrame]:
    adapter, anns, embedders, scaler0, X_folds, y_folds = initialize_ann_fusion_ensemble(X, y, seed=seed)

    perf_rows = []
    f1_per_fold = []

    for i in range(1, 10):
        Xi_raw = X_folds[i]
        yi = y_folds[i]

        y_pred, pos_score = predict_ann_fusion(
            adapter=adapter,
            embedders=embedders,
            scaler0=scaler0,
            X=Xi_raw,
        )

        acc = accuracy_score(yi, y_pred)
        prec = precision_score(yi, y_pred, zero_division=0)
        rec = recall_score(yi, y_pred, zero_division=0)
        f1 = f1_score(yi, y_pred, zero_division=0)

        f1_per_fold.append(float(f1))
        perf_rows.append({
            "Fold": i,
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
        })

        Xi_scaled = scaler0.transform(Xi_raw).astype(np.float32)
        Xi_c1, Xi_c2, Xi_c3 = split_channels(Xi_scaled)

        anns[0].fit(Xi_c1, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)
        anns[1].fit(Xi_c2, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)
        anns[2].fit(Xi_c3, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)

        Z1_tr = embedders[0].predict(Xi_c1, verbose=0)
        Z2_tr = embedders[1].predict(Xi_c2, verbose=0)
        Z3_tr = embedders[2].predict(Xi_c3, verbose=0)
        adapter.partial_fit(Z1_tr, Z2_tr, Z3_tr, yi)

    perf_df = pd.DataFrame(perf_rows)
    return float(np.mean(f1_per_fold)), f1_per_fold, perf_df


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
