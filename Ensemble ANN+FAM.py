import numpy as np
import pandas as pd
import os
from datetime import datetime   
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
Model = keras.Model
Input = keras.layers.Input
Dense = keras.layers.Dense
Adam = keras.optimizers.Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.utils import shuffle
from functools import partial
from scipy.stats import mode

l1_norm = partial(np.linalg.norm, ord=1, axis=-1)

# -------------------------------
# Data loading and preprocessing
# -------------------------------
df = pd.read_csv("~/Desktop/spambase.dat.csv")

X = df.drop(columns=["target"]).values  
y = df["target"].astype(int).values

X, y = shuffle(X, y, random_state=42)

X_folds = np.array_split(X, 10)
y_folds = np.array_split(y, 10)

# -------------------------------
# ANN for Feature Extraction
# -------------------------------
def build_ann(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    features = Dense(4, activation='relu', name='feature_layer')(x)
    outputs = Dense(1, activation='sigmoid')(features)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ========================
# Fuzzy ART Implementation (Unsupervised)
# ========================
class FuzzyART:
    def __init__(self, alpha=1.0, rho=0, gamma=0.01, complement_coding=True):
        '''Initialize Fuzzy ART with parameters'''
        self.alpha = alpha  # Learning rate
        self.beta = 1 - alpha  # Complementary learning rate
        self.gamma = gamma  # Small constant to prevent division by zero
        self.rho = rho      # Vigilance parameter
        self.complement_coding = complement_coding # Whether to use complement coding
        self.w = None # Stores category weights

    def _init_weights(self, x):
        '''Weight initialization'''
        M = x.shape[0]
        # Initialize ARTa weights explicitly to 1
        self.w = np.ones((1, M))

    def _complement_code(self, x):
        '''Apply complement coding to input'''
        if self.complement_coding:
            return np.hstack((x, 1-x))
        else:
            return x
        
    def _add_category(self, x):
        '''Add a new cluster'''
        if self.w is None:
            self.w = np.atleast_2d(x)
        else:
            self.w = np.vstack((self.w, x))

    def _match_category(self, x):
        '''Find best matching category'''
        x = np.atleast_2d(x)  # Ensure x is the correct shape

        if self.w is None:
            return -1
        
        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))
        threshold = fuzzy_norm / l1_norm(x) >= self.rho
        if np.all(threshold == False):
            return -1
        else:
            return np.argmax(scores * threshold.astype(int))
        
    def train(self, x, epochs=1):
        '''Train the ART model'''
        samples = self._complement_code(np.atleast_2d(x))

        if self.w is None:
            self._init_weights(samples[0])

        for epoch in range(epochs):
            for sample in np.random.permutation(samples):
                category = self._match_category(sample)
                if category == -1:
                    self._add_category(sample)
                else:
                    w = self.w[category]
                    self.w[category] = (self.alpha * np.minimum(sample, w) +
                                        self.beta * w)
                    self.w = np.clip(self.w, 0, None)
        return self

    def predict(self, x):
        '''Predict categories for input samples'''
        samples = self._complement_code(np.atleast_2d(x))
        categories = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            categories[i] = self._match_category(sample)
        return categories

# ========================
# Fuzzy ARTMAP Implementation (Supervised)
# ========================
class FuzzyARTMAP(FuzzyART):
    def __init__(self, alpha=1.0, rho=0, gamma=0.01, epsilon=0.0001, 
                 complement_coding=True):
        
        self.alpha = alpha  # learning rate
        self.beta = 1 - alpha
        self.gamma = gamma  # choice parameter
        self.rho = rho  # vigilance
        self.epsilon = epsilon  # match tracking
        self.complement_coding = complement_coding
        self.w = None
        self.out_w = None
        self.n_classes = 0 

    def _init_weights(self, M):
        """Initialize weights and output weights"""
        self.w = np.ones((1, M))
        self.out_w = np.zeros((1, self.n_classes))

    def _complement_code(self, x):
        if self.complement_coding:
            return np.hstack((x, 1-x))
        else:
            return x

    def _add_category(self, x, y):
        """Add a new category and assign a label"""
        if self.w is None:
            self._init_weights(x.shape[0])
        else:
            self.w = np.vstack((self.w, x))
            self.out_w = np.vstack((self.out_w, np.zeros((1, self.n_classes))))
        self.out_w[-1, y] = 1

    def _match_category(self, x, y=None, return_scores=False):
        """Find best matching category with supervision"""
        x = np.atleast_2d(x) 
        if self.w is None:
            return -1
        _rho = self.rho
        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))
        norms = fuzzy_norm / l1_norm(x)
        threshold = norms >= _rho

        while not np.all(threshold == False):
            y_ = np.argmax(scores * threshold.astype(int))
            if y is None or self.out_w[y_, y] == 1:
                return (y_, scores) if return_scores else y_
            else:
                _rho = norms[y_] + self.epsilon
                norms[y_] = 0
                threshold = norms >= _rho
        return (-1, None) if return_scores else -1

    def train(self, x, y, epochs=1):
        '''Train the Fuzzy ARTMAP model with supervision'''
        samples = self._complement_code(np.atleast_2d(x))
        self.n_classes = len(set(y))

        if self.w is None:
            self._init_weights(samples.shape[1])

        idx = np.arange(len(samples), dtype=np.uint32) 

        for epoch in range(epochs):
            idx = np.random.permutation(idx)
            for sample, label in zip(samples[idx], y[idx]):
                category = self._match_category(sample, label)
                if category == -1:
                    self._add_category(sample, label)
                else:
                    w = self.w[category]
                    self.w[category] = (self.alpha * np.minimum(sample, w) +
                                        self.beta * w)
                    self.w = np.clip(self.w, 0, None)
        return self

    def predict(self, x):
        """Predict labels based on trained mapping"""
        samples = self._complement_code(np.atleast_2d(x))
        labels = np.zeros(len(samples))
        match_scores = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            category, scores = self._match_category(sample, return_scores=True)
            if category != -1:
                labels[i] = np.argmax(self.out_w[category])
                match_scores[i] = scores[category]
            else:
                labels[i] = -1
                match_scores[i] = 0
        return labels, match_scores

# -------------------------------
# Initialize 3 Ensemble Models
# -------------------------------
ensemble_models = []
for _ in range(3):
    ann_model = build_ann(X_folds[0].shape[1])
    fam_model = FuzzyARTMAP(alpha=1.0, rho=0)
    extractor = Model(inputs=ann_model.input, outputs=ann_model.get_layer("feature_layer").output)
    ensemble_models.append((ann_model, fam_model, extractor))

performance = []
fold_predictions = []
fold_confusions = []

# -------------------------------
# Train All Models on Fold 0
# -------------------------------
scaler_0 = StandardScaler().fit(X_folds[0])
X0_scaled = scaler_0.transform(X_folds[0])
y0 = y_folds[0]

for ann_model, fam_model, extractor in ensemble_models:
    ann_model.fit(X0_scaled, y0, epochs=20, batch_size=32, verbose=0)
    features_0 = extractor.predict(X0_scaled, verbose=0)
    global_minmax = MinMaxScaler().fit(features_0)
    features_0 = global_minmax.transform(features_0)
    fam_model.train(features_0, y0, epochs=1)

# -------------------------------
# Test + Train Online for Folds 1–9
# -------------------------------
for i in range(1, 10):
    print(f"\n🔁 Fold {i-1}→{i}: Testing before training")

    X_test_scaled = scaler_0.transform(X_folds[i])
    y_test = y_folds[i]
    all_preds = []
   
    test_features = extractor.predict(X_test_scaled, verbose=0)
    test_features = global_minmax.transform(test_features)
    all_preds = np.stack([fam_model.predict(test_features)[0] 
        for _, fam_model, _ in ensemble_models], axis=0)
    all_scores = np.stack([fam_model.predict(test_features)[1] 
        for _, fam_model, _ in ensemble_models], axis=0)

    majority_vote   = mode(all_preds, axis=0).mode.flatten()
    ensemble_score  = all_scores.mean(axis=0)

    # Metrics
    acc = accuracy_score(y_test, majority_vote)
    prec = precision_score(y_test, majority_vote)
    rec = recall_score(y_test, majority_vote)
    f1 = f1_score(y_test, majority_vote)
    auc = roc_auc_score(y_test, ensemble_score) 
    mcc = matthews_corrcoef(y_test, majority_vote)
    cm = confusion_matrix(y_test, majority_vote)
    tn, fp, fn, tp = cm.ravel()

    performance.append({
        "Fold": f"{i-1}→{i}",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUROC": auc,
        "MCC": mcc,
        "TP": cm[1,1],
        "FP": cm[0,1],
        "TN": cm[0,0],
        "FN": cm[1,0]
    })

    # Save predictions
    pred_df = pd.DataFrame({
        "True Label": y_test,
        "Predicted (Majority Vote)": majority_vote
    })
    fold_predictions.append((f"Fold_{i}_Predictions", pred_df))

    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    fold_confusions.append((f"Fold_{i}_ConfusionMatrix", cm_df))

    # 🔧 Online Training (Next Fold)
    for ann_model, fam_model, extractor in ensemble_models:
        ann_model.fit(X_test_scaled, y_test, epochs=20, batch_size=32, verbose=0)
        train_features = extractor.predict(X_test_scaled, verbose=0)
        train_features = global_minmax.transform(train_features)
        fam_model.train(train_features, y_test, epochs=1)

    # Save F2 node weights for this model and fold
    weights_df = pd.DataFrame(fam_model.w)
    weights_df.index.name = f"F2 Nodes (Model)"
    weights_df.columns = [f"Feature {j}" for j in range(weights_df.shape[1])]

    # Initialize list if not exists
    if 'f2_weights_per_fold' not in locals():
        f2_weights_per_fold = [[] for _ in range(9)]  # 9 folds (1–9)

    f2_weights_per_fold[i-1].append(weights_df)  # Store for this fold and model

# -------------------------------
# Display Results
# -------------------------------
perf_df = pd.DataFrame(performance)
print("\n📊 Fold-by-Fold Ensemble Performance:")
print(perf_df.round(4))

# Save results to Excel
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"Hybrid_ANN_FAM_Ensemble_Online_Performance_{timestamp}.xlsx"
output_path = os.path.expanduser(os.path.join("~/Desktop", filename))

with pd.ExcelWriter(output_path) as writer:
    # Save metrics
    perf_df.to_excel(writer, index=False, sheet_name='Performance Metrics')

    # Save F2 weights
    for fold_idx, fold_weights in enumerate(f2_weights_per_fold, start=1):
        for model_idx, df in enumerate(fold_weights, start=1):
            df.to_excel(writer, sheet_name=f"Fold_{fold_idx}_Model_{model_idx}")

    # Save per-fold predictions
    for sheet_name, df in fold_predictions:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    # Save confusion matrices
    for sheet_name, df in fold_confusions:
        df.to_excel(writer, sheet_name=sheet_name)
