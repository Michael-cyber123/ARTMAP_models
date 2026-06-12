import os
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import f1_score
from scipy.stats import mode
from fuzzy_art import FuzzyARTMAP

# =========================================================
# CONFIG
# =========================================================
EMBED_DIM = 4
EPOCHS_FOLD0 = 20
EPOCHS_ONLINE = 20
BATCH_SIZE = 32
ANN_LR = 1e-3
N_MODELS = 3

# =========================================================
# UTILS
# =========================================================
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_xy(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    y = df.iloc[:, -1].astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    return X, y


# =========================================================
# ANN FEATURE EXTRACTOR
# =========================================================
def build_ann(input_dim: int) -> Tuple[keras.Model, keras.Model]:
    inp = L.Input(shape=(input_dim,))
    x = L.Dense(64, activation="relu")(inp)
    x = L.Dense(32, activation="relu")(x)
    z = L.Dense(EMBED_DIM, activation="relu", name="feature_layer")(x)
    out = L.Dense(1, activation="sigmoid")(z)

    ann = keras.Model(inp, out)
    ann.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ANN_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    embedder = keras.Model(inp, ann.get_layer("feature_layer").output)
    return ann, embedder


# =========================================================
# ENSEMBLE INITIALIZATION
# =========================================================
def initialize_ann_fam_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    fam_alpha: float = 1.0,
    fam_rho: float = 0.0,
    fam_gamma: float = 0.01,
    fam_epsilon: float = 1e-4,) 
    Tuple[
    List[Tuple[keras.Model, keras.Model, FuzzyARTMAP]],
    StandardScaler,
    MinMaxScaler,
    int,
    List[np.ndarray],
    List[np.ndarray],
]:
    
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

    input_scaler = StandardScaler().fit(X0)
    X0s = input_scaler.transform(X0)

    ensemble: List[Tuple[keras.Model, keras.Model, FuzzyARTMAP]] = []
    fold0_features = []

    for _ in range(N_MODELS):
        ann, embedder = build_ann(X0.shape[1])

        fam = FuzzyARTMAP(
            alpha=fam_alpha,
            rho=fam_rho,
            gamma=fam_gamma,
            epsilon=fam_epsilon,
            complement_coding=True,
        )

        ann.fit(X0s, y0, epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)
        f0 = embedder.predict(X0s, verbose=0)
        fold0_features.append(f0)
        ensemble.append((ann, embedder, fam))

    all_f0 = np.vstack(fold0_features)
    feature_scaler = MinMaxScaler().fit(all_f0)

    for (_, _, fam), f0 in zip(ensemble, fold0_features):
        fam.train(feature_scaler.transform(f0), y0, epochs=1)

    vals, counts = np.unique(y0, return_counts=True)
    fallback_label = int(vals[np.argmax(counts)])

    return ensemble, input_scaler, feature_scaler, fallback_label, X_folds, y_folds


# =========================================================
# ENSEMBLE PREDICTION
# =========================================================
def predict_ann_fam_ensemble(
    ensemble: List[Tuple[keras.Model, keras.Model, FuzzyARTMAP]],
    input_scaler: StandardScaler,
    feature_scaler: MinMaxScaler,
    X: np.ndarray,
    fallback_label: int,)
    Tuple[np.ndarray, np.ndarray]:
    Xs = input_scaler.transform(X)

    all_preds = []
    all_scores = []

    for ann, embedder, fam in ensemble:
        feats = embedder.predict(Xs, verbose=0)
        feats = feature_scaler.transform(feats)

        pred, score = fam.predict(feats)

        if REPLACE_NO_MATCH_WITH_FOLD0_MODE:
            pred = np.where(pred == -1, fallback_label, pred).astype(int)

        all_preds.append(pred)
        all_scores.append(score)

    all_preds = np.stack(all_preds, axis=0)
    all_scores = np.stack(all_scores, axis=0)

    majority = mode(all_preds, axis=0).mode.flatten()
    mean_scores = np.mean(all_scores, axis=0)

    return majority, mean_scores


# =========================================================
# ONLINE EVALUATION
# =========================================================
def run_online_ann_fam_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    fam_alpha: float = 1.0,
    fam_rho: float = 0.0,
    fam_gamma: float = 0.01,
    fam_epsilon: float = 1e-4,) 
    Tuple[float, List[float]]:
   
    (
        ensemble,
        input_scaler,
        feature_scaler,
        fallback_label,
        X_folds,
        y_folds,
    ) = initialize_ann_fam_ensemble(
        X,
        y,
        seed=seed,
        fam_alpha=fam_alpha,
        fam_rho=fam_rho,
        fam_gamma=fam_gamma,
        fam_epsilon=fam_epsilon,
    )

    f1_per_fold = []

    for i in range(1, 10):
        Xi_raw = X_folds[i]
        yi = y_folds[i]

        maj_pred, _ = predict_ann_fam_ensemble(
            ensemble=ensemble,
            input_scaler=input_scaler,
            feature_scaler=feature_scaler,
            X=Xi_raw,
            fallback_label=fallback_label,
        )

        f1_per_fold.append(float(f1_score(yi, maj_pred, zero_division=0)))

        Xi_scaled = input_scaler.transform(Xi_raw)

        for ann, embedder, fam in ensemble:
            ann.fit(Xi_scaled, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)
            tr = embedder.predict(Xi_scaled, verbose=0)
            fam.train(feature_scaler.transform(tr), yi, epochs=1)

    return float(np.mean(f1_per_fold)), f1_per_fold


# =========================================================
# EXAMPLE
# =========================================================
def main():
    csv_path = os.path.expanduser("")
    X, y = load_xy(csv_path)

    mean_f1, f1_per_fold = run_online_ann_fam_ensemble(
        X=X,
        y=y,
        seed=42,
        fam_alpha=1.0,
        fam_rho=0.0,
        fam_gamma=0.01,
        fam_epsilon=1e-4,
    )

    print("ANN+FAM ensemble")
    print("Mean F1:", round(mean_f1, 4))
    print("F1 per fold:", [round(v, 4) for v in f1_per_fold])


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
