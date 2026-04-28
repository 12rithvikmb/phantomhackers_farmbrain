"""
FarmBrain Crop Recommendation Engine
RandomForest with fallback to rule-based prediction.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = logging.getLogger("farmbrain.crop_engine")

# Rule-based fallback thresholds
RULE_BASED_CROPS = {
    "rice":       {"rainfall": (150, 999), "temperature": (20, 35), "humidity": (60, 100)},
    "wheat":      {"rainfall": (50, 150),  "temperature": (10, 25), "humidity": (40, 70)},
    "maize":      {"rainfall": (50, 200),  "temperature": (18, 35), "humidity": (50, 80)},
    "cotton":     {"rainfall": (50, 150),  "temperature": (25, 40), "humidity": (40, 70)},
    "sugarcane":  {"rainfall": (100, 300), "temperature": (20, 38), "humidity": (65, 100)},
    "chickpea":   {"rainfall": (30, 100),  "temperature": (15, 30), "humidity": (30, 65)},
    "mungbean":   {"rainfall": (60, 150),  "temperature": (25, 40), "humidity": (50, 80)},
    "coffee":     {"rainfall": (150, 300), "temperature": (15, 28), "humidity": (70, 100)},
    "banana":     {"rainfall": (100, 300), "temperature": (20, 35), "humidity": (70, 100)},
    "mango":      {"rainfall": (50, 200),  "temperature": (24, 38), "humidity": (50, 80)},
}

FEATURES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]


class CropRecommendationEngine:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_cols = FEATURES
        self.accuracy = 0.0
        self.mode = "untrained"

    def train(self, df: pd.DataFrame) -> bool:
        """Train RandomForest; fallback to reduced features if needed."""
        logger.info("[TRAIN] Starting crop recommendation model training...")

        try:
            return self._train_full(df)
        except Exception as e:
            logger.error(f"[ERROR] Full training failed: {e}")
            logger.warning("[FALLBACK] Retrying with reduced feature set...")
            try:
                return self._train_reduced(df)
            except Exception as e2:
                logger.error(f"[ERROR] Reduced training also failed: {e2}")
                logger.warning("[FALLBACK] Switching to rule-based mode")
                self.mode = "rule-based"
                return False

    def _train_full(self, df: pd.DataFrame) -> bool:
        available = [c for c in FEATURES if c in df.columns]
        if "label" not in df.columns:
            raise ValueError("Missing 'label' column in crop dataset")
        if len(available) < 3:
            raise ValueError(f"Too few features: {available}")

        X = df[available].values
        y = self.label_encoder.fit_transform(df["label"].values)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        self.accuracy = accuracy_score(y_test, self.model.predict(X_test))
        self.feature_cols = available
        self.is_trained = True
        self.mode = "ml"
        logger.info(f"[TRAIN] Model trained. Accuracy: {self.accuracy:.3f}. Features: {available}")
        return True

    def _train_reduced(self, df: pd.DataFrame) -> bool:
        reduced = ["temperature", "humidity", "rainfall"]
        available = [c for c in reduced if c in df.columns]
        if len(available) < 2:
            raise ValueError("Not enough features for reduced training")

        X = df[available].values
        y = self.label_encoder.fit_transform(df["label"].values)
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X, y)

        self.feature_cols = available
        self.is_trained = True
        self.mode = "ml-reduced"
        logger.info(f"[TRAIN] Reduced model trained. Features: {available}")
        return True

    def predict(self, N: float, P: float, K: float, temperature: float,
                humidity: float, ph: float, rainfall: float) -> dict:
        """Return top 3 crop recommendations with confidence scores."""
        input_map = {"n": N, "p": P, "k": K, "temperature": temperature,
                     "humidity": humidity, "ph": ph, "rainfall": rainfall}

        if self.is_trained and self.model is not None:
            try:
                X = np.array([[input_map.get(f, 0.0) for f in self.feature_cols]])
                proba = self.model.predict_proba(X)[0]
                top3_idx = np.argsort(proba)[::-1][:3]
                top_crops = [self.label_encoder.classes_[i] for i in top3_idx]
                top_conf = [round(float(proba[i]) * 100, 2) for i in top3_idx]
                return {
                    "best_crop": top_crops[0],
                    "top_crops": top_crops,
                    "confidence": top_conf,
                    "mode": self.mode,
                    "status": "success",
                }
            except Exception as e:
                logger.error(f"[ERROR] ML predict failed: {e}, falling back to rule-based")

        return self._rule_based_predict(temperature, humidity, rainfall)

    def _rule_based_predict(self, temperature: float, humidity: float, rainfall: float) -> dict:
        logger.info("[FALLBACK] Using rule-based crop prediction")
        scores = {}
        for crop, rules in RULE_BASED_CROPS.items():
            score = 0
            r_lo, r_hi = rules["rainfall"]
            t_lo, t_hi = rules["temperature"]
            h_lo, h_hi = rules["humidity"]
            if r_lo <= rainfall <= r_hi: score += 3
            if t_lo <= temperature <= t_hi: score += 3
            if h_lo <= humidity <= h_hi: score += 2
            scores[crop] = score

        sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_crops[:3]
        total = sum(s for _, s in top3) or 1
        top_crops = [c for c, _ in top3]
        top_conf = [round(s / total * 100, 2) for _, s in top3]

        return {
            "best_crop": top_crops[0],
            "top_crops": top_crops,
            "confidence": top_conf,
            "mode": "rule-based",
            "status": "fallback used",
        }
