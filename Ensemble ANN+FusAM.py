import os
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)
from sklearn.utils import shuffle as sk_shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from fusion_artmap import FusionARTMAP
from fuzzy_art import complement_code

DATASETS = [
    (0,  os.path.expanduser("~/Desktop/Datasets/spambase.dat.csv"))
]
OUT_DIR = os.path.expanduser("~/Desktop/ann3_spam_results")

SLICE_WORD = slice(0, 48)
SLICE_CHAR = slice(48, 54)
SLICE_CAPS = slice(54, 57)

EMBED_DIM = 4
EPOCHS_FOLD0 = 20
EPOCHS_ONLINE = 20
BATCH_SIZE = 32
ANN_LR = 1e-3

FAM_ALPHA  = 0.01
FAM_BETA   = 1.0
FAM_RHO_C  = 0.0
FAM_RHO_A  = 0.0
FAM_RHO_B  = 1.0
FAM_RHO_AB = 1.0
FAM_EPS    = 0.0001
FAM_MAX_PMT_ITERS = 10
FAM_ARTA_COMPLEMENT = False
ENSEMBLE_SIZE = 3
RUNS = 10  

# ======================= UTILS =======================
def set_all_seeds(seed: int):
    import random, os
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def short_sheet(name: str, maxlen: int = 31) -> str:
    bad = {":", "\\", "/", "?", "*", "[", "]"}
    cleaned = "".join(ch if ch not in bad else "-" for ch in str(name))
    return cleaned[:maxlen]

def load_xy(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)
    y = df.iloc[:, -1].astype(int).values
    X = df.iloc[:, :-1].values
    return X, y

def build_ann(input_dim: int, embed_dim: int = 4, lr: float = 1e-3, name: str = "c1"):
    x = L.Input(shape=(input_dim,), name=f"{name}_in")
    h = L.Dense(64, activation="relu", name=f"{name}_h1")(x)
    h = L.Dense(32, activation="relu", name=f"{name}_h2")(h)
    z = L.Dense(embed_dim, activation="relu", name=f"{name}_z")(h)
    yhat = L.Dense(1, activation="sigmoid", name=f"{name}_y")(z)
    model = keras.Model(x, yhat, name=f"ann_{name}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    embedder = keras.Model(x, z, name=f"embedder_{name}")
    return model, embedder

class FusionAdapter:
    """
    Applies GLOBAL MinMax fitted on Fold-0 (across Z=[Z1|Z2|Z3]), clips to [0,1],
    and exposes predict_with_scores() for AUROC.
    """
    def __init__(self, fam: FusionARTMAP, mins: np.ndarray, maxs: np.ndarray,
                 idx1: slice, idx2: slice, idx3: slice):
        self.fam  = fam
        self.mins = mins
        self.maxs = maxs
        self.idx1 = idx1
        self.idx2 = idx2
        self.idx3 = idx3

    def _minmax(self, Z: np.ndarray, sl: slice) -> np.ndarray:
        return np.clip((Z - self.mins[sl]) / (self.maxs[sl] - self.mins[sl] + 1e-12), 0.0, 1.0)

    def _prep_batch(self, Z1: np.ndarray, Z2: np.ndarray, Z3: np.ndarray):
        return self._minmax(Z1, self.idx1), self._minmax(Z2, self.idx2), self._minmax(Z3, self.idx3)

    def partial_fit(self, Z1: np.ndarray, Z2: np.ndarray, Z3: np.ndarray, y: np.ndarray):
        Z1, Z2, Z3 = self._prep_batch(Z1, Z2, Z3)
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
        return self._predict_with_scores_fixed(Z1, Z2, Z3)

    def _predict_with_scores_fixed(self, Z1: np.ndarray, Z2: np.ndarray, Z3: np.ndarray):
        n = Z1.shape[0]
        y_pred = np.zeros(n, dtype=int)
        pos_score = np.zeros(n, dtype=float)
        for i in range(n):
            y_pred[i], pos_score[i] = self._predict_one_with_score(Z1[i], Z2[i], Z3[i])
        return y_pred, pos_score

def majority_vote(preds_2d: np.ndarray) -> np.ndarray:
    def vote(col):
        bc = np.bincount(col.astype(int), minlength=2)
        return int(np.argmax(bc))
    return np.apply_along_axis(vote, axis=0, arr=preds_2d)

# ===================== ONE EXPERIMENT (ENSEMBLE) =====================
def run_one_experiment(X: np.ndarray, y: np.ndarray, seed: int) -> pd.DataFrame:
    set_all_seeds(seed)

    X_shuf, y_shuf = sk_shuffle(X, y, random_state=seed)
    X_folds = np.array_split(X_shuf, 10)
    y_folds = np.array_split(y_shuf, 10)

    scaler0 = StandardScaler().fit(X_folds[0])
    X0s = scaler0.transform(X_folds[0]).astype("float32")
    X0_c1, X0_c2, X0_c3 = X0s[:, SLICE_WORD], X0s[:, SLICE_CHAR], X0s[:, SLICE_CAPS]
    y0 = y_folds[0]

    ensemble: List[Dict] = []
    fold0_concat_list = []

    for k in range(ENSEMBLE_SIZE):
        tf.random.set_seed(seed + 1000 + k)

        ann1, emb1 = build_ann(X0_c1.shape[1], EMBED_DIM, ANN_LR, name=f"m{k}_c1")
        ann2, emb2 = build_ann(X0_c2.shape[1], EMBED_DIM, ANN_LR, name=f"m{k}_c2")
        ann3, emb3 = build_ann(X0_c3.shape[1], EMBED_DIM, ANN_LR, name=f"m{k}_c3")

        ann1.fit(X0_c1, y0, epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)
        ann2.fit(X0_c2, y0, epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)
        ann3.fit(X0_c3, y0, epochs=EPOCHS_FOLD0, batch_size=BATCH_SIZE, verbose=0)

        Z1_0 = emb1.predict(X0_c1, verbose=0)
        Z2_0 = emb2.predict(X0_c2, verbose=0)
        Z3_0 = emb3.predict(X0_c3, verbose=0)
        Z0_concat = np.concatenate([Z1_0, Z2_0, Z3_0], axis=1)
        fold0_concat_list.append(Z0_concat)

        ensemble.append({
            "ann1": ann1, "emb1": emb1,
            "ann2": ann2, "emb2": emb2,
            "ann3": ann3, "emb3": emb3,
            "Z1_0": Z1_0, "Z2_0": Z2_0, "Z3_0": Z3_0
        })

    pooled0 = np.vstack(fold0_concat_list) 
    mins = pooled0.min(axis=0)
    maxs = pooled0.max(axis=0)

    s1 = EMBED_DIM
    s2 = EMBED_DIM
    s3 = EMBED_DIM
    idx1 = slice(0, s1); idx2 = slice(s1, s1+s2); idx3 = slice(s1+s2, s1+s2+s3)

    for m in ensemble:
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
        adapter = FusionAdapter(fam_core, mins, maxs, idx1, idx2, idx3)
        adapter.partial_fit(m["Z1_0"], m["Z2_0"], m["Z3_0"], y0)

        m["fam"] = fam_core
        m["adapter"] = adapter
        
        del m["Z1_0"], m["Z2_0"], m["Z3_0"]

    rows = []
    for i in range(1, 10):
        Xi = scaler0.transform(X_folds[i]).astype("float32")
        yi = y_folds[i]

        Xi_c1, Xi_c2, Xi_c3 = Xi[:, SLICE_WORD], Xi[:, SLICE_CHAR], Xi[:, SLICE_CAPS]

       
        preds_list = []
        scores_list = []
        for m in ensemble:
            Z1 = m["emb1"].predict(Xi_c1, verbose=0)
            Z2 = m["emb2"].predict(Xi_c2, verbose=0)
            Z3 = m["emb3"].predict(Xi_c3, verbose=0)
            y_pred, pos_score = m["adapter"]._predict_with_scores_fixed(Z1, Z2, Z3)
            preds_list.append(y_pred.astype(int))
            scores_list.append(pos_score.astype(float))

        preds = np.stack(preds_list, axis=0)      
        scores = np.stack(scores_list, axis=0)    

        y_pred_ens = majority_vote(preds)       
        pos_score_ens = scores.mean(axis=0)       

        acc  = accuracy_score(yi, y_pred_ens)
        prec = precision_score(yi, y_pred_ens, zero_division=0)
        rec  = recall_score(yi, y_pred_ens, zero_division=0)
        f1   = f1_score(yi, y_pred_ens, zero_division=0)
        try:
            auc = roc_auc_score(yi, pos_score_ens)
        except Exception:
            auc = float("nan")
        mcc  = matthews_corrcoef(yi, y_pred_ens)
        cm   = confusion_matrix(yi, y_pred_ens, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Run": seed - 41,  
            "Fold": i,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "AUROC": auc, "MCC": mcc, "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })

        for m in ensemble:
            m["ann1"].fit(Xi_c1, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)
            m["ann2"].fit(Xi_c2, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)
            m["ann3"].fit(Xi_c3, yi, epochs=EPOCHS_ONLINE, batch_size=BATCH_SIZE, verbose=0)

            Z1_tr = m["emb1"].predict(Xi_c1, verbose=0)
            Z2_tr = m["emb2"].predict(Xi_c2, verbose=0)
            Z3_tr = m["emb3"].predict(Xi_c3, verbose=0)
            m["adapter"].partial_fit(Z1_tr, Z2_tr, Z3_tr, yi)

    return pd.DataFrame(rows)

# =========================== MAIN ============================
if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    os.makedirs(OUT_DIR, exist_ok=True)
    TS = datetime.now().strftime("%Y%m%d_%H%M%S")

    metric_cols = ["Accuracy","Precision","Recall","F1","AUROC","MCC","TP","FP","TN","FN"]
    main_metrics = ["Accuracy","Precision","Recall","F1","AUROC","MCC"]  # format with CI

    for noise, csv_path in DATASETS:
        print(f"\n=== Noise {noise}: {csv_path} ===")
        X, y = load_xy(csv_path)
        assert X.shape[1] >= 57, f"Expected at least 57 features (got {X.shape[1]})."

        run_dfs = []
        for r in range(RUNS):
            seed = 42 + r
            print(f"  • Run {r+1}/{RUNS} (seed={seed})")
            df = run_one_experiment(X, y, seed=seed)
            run_dfs.append(df)

        out_path = os.path.join(OUT_DIR, f"ANN3_Fusion_10Runs_Noise{noise}_{TS}.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            for r, df in enumerate(run_dfs, start=1):
                df.to_excel(writer, index=False, sheet_name=short_sheet(f"Run_{r:02d}"))

            combined = pd.concat(run_dfs, ignore_index=True)
            combined.to_excel(writer, index=False, sheet_name=short_sheet("All_10_Runs"))

            avg_per_fold = (combined
                            .groupby("Fold", as_index=False)[metric_cols]
                            .mean(numeric_only=True))
            avg_per_fold.to_excel(writer, index=False, sheet_name=short_sheet("Average_Performance"))

            summary_row = {"Noise": noise, "Method": "ann3_fusion"}
            for m in main_metrics:
                s = combined[m].dropna()
                if len(s) > 1:
                    mean = float(s.mean())
                    std  = float(s.std(ddof=1))
                    n    = int(s.count())
                    ci   = 1.96 * std / np.sqrt(n)
                    summary_row[m] = f"{mean:.4f}±{ci:.4f} (std±{std:.4f})"
                elif len(s) == 1:
                    summary_row[m] = f"{float(s.iloc[0]):.4f}±nan (std±nan)"
                else:
                    summary_row[m] = "nan±nan (std±nan)"

            for m in ["TP","FP","TN","FN"]:
                s = combined[m].dropna()
                if len(s) >= 1:
                    mean = float(s.mean())
                    std  = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
                    summary_row[m] = f"{mean:.2f}±{std:.2f}" if not np.isnan(std) else f"{mean:.2f}±nan"
                else:
                    summary_row[m] = "nan±nan"

            overall_df = pd.DataFrame([summary_row],
                                      columns=["Noise","Method"]+metric_cols)
            overall_df.to_excel(writer, index=False, sheet_name=short_sheet("Overall_Summary"))

        print(f"📁 Wrote: {out_path}")
