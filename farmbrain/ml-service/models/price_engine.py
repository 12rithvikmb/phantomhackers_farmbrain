"""
FarmBrain Price Prediction Engine
ARIMA with moving-average fallback.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("farmbrain.price_engine")


class PricePredictionEngine:
    def __init__(self):
        self.price_data = {}   # crop -> list of modal prices
        self.arima_models = {} # crop -> fitted ARIMA
        self.is_ready = False

    def load_data(self, df: pd.DataFrame):
        logger.info("[PRICE] Loading market price data...")
        if df.empty:
            logger.warning("[PRICE] Empty price dataframe; using static defaults")
            return

        df = df.copy()
        for col in ["min_price", "max_price", "modal_price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "modal_price" not in df.columns or "crop" not in df.columns:
            logger.warning("[PRICE] Required columns missing; skipping price load")
            return

        df = df.dropna(subset=["modal_price", "crop"])
        for crop, group in df.groupby("crop"):
            prices = group["modal_price"].tolist()
            if len(prices) >= 4:
                self.price_data[crop.lower()] = prices

        logger.info(f"[PRICE] Loaded prices for {len(self.price_data)} crops")
        self._train_arima_models()
        self.is_ready = True

    def _train_arima_models(self):
        """Attempt ARIMA training for each crop; skip on failure."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            trained = 0
            for crop, prices in self.price_data.items():
                try:
                    if len(prices) < 8:
                        continue
                    model = ARIMA(prices, order=(2, 1, 2))
                    result = model.fit()
                    self.arima_models[crop] = result
                    trained += 1
                except Exception as e:
                    logger.debug(f"[PRICE] ARIMA failed for {crop}: {e}")
            logger.info(f"[PRICE] ARIMA trained for {trained} crops")
        except ImportError:
            logger.warning("[PRICE] statsmodels unavailable; ARIMA disabled")

    def predict(self, crop: str, months_ahead: int = 3) -> dict:
        crop_key = crop.lower()
        logger.info(f"[PRICE] Predicting price for crop='{crop}' months_ahead={months_ahead}")

        # Try ARIMA
        if crop_key in self.arima_models:
            try:
                result = self.arima_models[crop_key]
                forecast = result.forecast(steps=months_ahead)
                predicted = [max(0, round(float(v), 2)) for v in forecast]
                current = self.price_data[crop_key][-1] if self.price_data.get(crop_key) else predicted[0]
                return self._build_response(crop, current, predicted, "arima")
            except Exception as e:
                logger.warning(f"[PRICE] ARIMA predict failed for {crop}: {e}")

        # Fallback: moving average
        if crop_key in self.price_data:
            return self._moving_average_predict(crop, crop_key, months_ahead)

        # Last resort: static estimate
        return self._static_estimate(crop, months_ahead)

    def _moving_average_predict(self, crop: str, crop_key: str, months_ahead: int) -> dict:
        logger.info(f"[PRICE] Using moving average for {crop}")
        prices = self.price_data[crop_key]
        window = min(6, len(prices))
        ma = float(np.mean(prices[-window:]))
        trend = float(np.polyfit(range(window), prices[-window:], 1)[0])
        predicted = [round(max(0, ma + trend * i), 2) for i in range(1, months_ahead + 1)]
        return self._build_response(crop, prices[-1], predicted, "moving-average")

    def _static_estimate(self, crop: str, months_ahead: int) -> dict:
        logger.warning(f"[PRICE] No data for {crop}; returning static fallback")
        base = 2000.0
        predicted = [round(base * (1 + 0.02 * i), 2) for i in range(1, months_ahead + 1)]
        return self._build_response(crop, base, predicted, "static-fallback")

    def _build_response(self, crop: str, current: float, predicted: list, mode: str) -> dict:
        change_pct = round((predicted[-1] - current) / max(current, 1) * 100, 2)
        trend = "rising" if change_pct > 2 else "falling" if change_pct < -2 else "stable"
        return {
            "crop": crop,
            "current_price": round(current, 2),
            "predicted_prices": predicted,
            "price_trend": trend,
            "change_percent": change_pct,
            "mode": mode,
            "status": "success" if mode == "arima" else "fallback used",
        }

    def calculate_price_volatility(self, crop: str) -> float:
        """Return coefficient of variation (%) as volatility metric."""
        crop_key = crop.lower()
        if crop_key not in self.price_data or len(self.price_data[crop_key]) < 2:
            return 15.0  # safe default
        prices = np.array(self.price_data[crop_key])
        cv = float(np.std(prices) / np.mean(prices) * 100) if np.mean(prices) > 0 else 15.0
        return round(cv, 2)
