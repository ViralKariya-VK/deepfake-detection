import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize


class AnomalyDetector:
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.model         = None

    def fit(self, embeddings):
        print(f"Fitting Isolation Forest on {len(embeddings)} embeddings...")
        embeddings_norm = normalize(embeddings, norm="l2")
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(embeddings_norm)
        print("Fitting complete.")

    def score(self, embeddings):
        embeddings_norm = normalize(embeddings, norm="l2")
        raw_scores = self.model.decision_function(embeddings_norm)

        if len(raw_scores) == 1:
            anomaly_scores = 1 - np.clip(raw_scores + 0.5, 0, 1)
        else:
            anomaly_scores = 1 - (raw_scores - raw_scores.min()) / \
                                (raw_scores.max() - raw_scores.min())

        return anomaly_scores

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"Model loaded from: {path}")
