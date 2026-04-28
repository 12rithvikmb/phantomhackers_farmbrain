"""
FarmBrain Risk Analysis, Profit Estimator, and Timeline Generator
All engines are self-healing and always return a value.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("farmbrain.analytics")


# ─────────────────────────────────────────────
# RISK ANALYSIS ENGINE
# ─────────────────────────────────────────────

CROP_RISK_BASE = {
    "rice": "Medium", "wheat": "Low", "maize": "Low", "cotton": "High",
    "sugarcane": "Medium", "chickpea": "Low", "mungbean": "Low",
    "coffee": "High", "banana": "Medium", "mango": "Medium",
    "pomegranate": "High", "grapes": "High", "papaya": "Medium",
}


class RiskAnalysisEngine:
    def __init__(self):
        self.weather_df = pd.DataFrame()

    def load_weather(self, df: pd.DataFrame):
        self.weather_df = df.copy()
        logger.info(f"[RISK] Loaded weather data: {len(df)} rows")

    def analyze(self, crop: str, rainfall: float, price_volatility: float) -> dict:
        logger.info(f"[RISK] Analyzing risk for crop='{crop}'")
        try:
            return self._compute_risk(crop, rainfall, price_volatility)
        except Exception as e:
            logger.error(f"[RISK] Risk computation failed: {e}")
            return self._safe_default(crop)

    def _compute_risk(self, crop: str, rainfall: float, price_volatility: float) -> dict:
        risk_score = 0.0
        factors = []

        # Rainfall variance from historical data
        rainfall_variance = self._get_rainfall_variance()
        if rainfall_variance > 30:
            risk_score += 35
            factors.append("High rainfall variability")
        elif rainfall_variance > 15:
            risk_score += 20
            factors.append("Moderate rainfall variability")
        else:
            risk_score += 5
            factors.append("Stable rainfall pattern")

        # Price volatility
        if price_volatility > 25:
            risk_score += 35
            factors.append("High price volatility")
        elif price_volatility > 12:
            risk_score += 20
            factors.append("Moderate price volatility")
        else:
            risk_score += 8
            factors.append("Stable market prices")

        # Crop-specific base risk
        base = CROP_RISK_BASE.get(crop.lower(), "Medium")
        if base == "High":
            risk_score += 20
        elif base == "Medium":
            risk_score += 10

        # Rainfall deviation from optimal
        optimal = 150.0
        dev = abs(rainfall - optimal) / optimal * 100
        if dev > 50:
            risk_score += 15
            factors.append("Rainfall far from optimal")

        risk_level = "Low" if risk_score < 35 else "Medium" if risk_score < 65 else "High"

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 1),
            "risk_factors": factors,
            "mitigation": self._mitigation_advice(risk_level, factors),
            "status": "success",
        }

    def _get_rainfall_variance(self) -> float:
        if self.weather_df.empty or "rainfall" not in self.weather_df.columns:
            return 20.0  # safe medium default
        vals = self.weather_df["rainfall"].dropna()
        if len(vals) < 5:
            return 20.0
        return float(np.std(vals))

    def _mitigation_advice(self, risk: str, factors: list) -> list:
        advice = []
        if risk == "High":
            advice += ["Consider crop insurance", "Diversify crop selection", "Install irrigation system"]
        elif risk == "Medium":
            advice += ["Monitor weather forecasts weekly", "Set aside 20% buffer funds"]
        else:
            advice += ["Maintain current practices", "Document successful methods"]
        if any("rainfall" in f.lower() for f in factors):
            advice.append("Invest in water conservation techniques")
        return advice

    def _safe_default(self, crop: str) -> dict:
        logger.warning("[RISK] Returning safe default risk: Medium")
        return {
            "risk_level": "Medium",
            "risk_score": 40.0,
            "risk_factors": ["Data insufficient for full analysis"],
            "mitigation": ["Monitor market prices", "Consult local agricultural office"],
            "status": "fallback used",
        }


# ─────────────────────────────────────────────
# PROFIT ESTIMATOR
# ─────────────────────────────────────────────

DEFAULT_YIELD_PER_HECTARE = {
    "rice": 3.5, "wheat": 3.2, "maize": 4.0, "cotton": 1.8, "sugarcane": 60.0,
    "chickpea": 1.2, "mungbean": 1.0, "coffee": 0.8, "banana": 25.0, "mango": 8.0,
    "pomegranate": 12.0, "grapes": 15.0, "papaya": 40.0, "lentil": 1.1,
    "kidneybeans": 1.3, "blackgram": 0.9, "pigeonpeas": 1.4, "coconut": 10.0,
    "jute": 2.5, "orange": 10.0, "apple": 7.0, "watermelon": 20.0, "muskmelon": 15.0,
}

DEFAULT_COST_PER_HECTARE = {
    "rice": 25000, "wheat": 22000, "maize": 18000, "cotton": 30000, "sugarcane": 40000,
    "chickpea": 15000, "coffee": 35000, "banana": 28000, "mango": 22000, "grapes": 45000,
}


class ProfitEstimator:
    def __init__(self):
        self.production_df = pd.DataFrame()

    def load_production(self, df: pd.DataFrame):
        self.production_df = df.copy()
        logger.info(f"[PROFIT] Loaded production data: {len(df)} rows")

    def estimate(self, crop: str, area_hectares: float, price_per_ton: float) -> dict:
        logger.info(f"[PROFIT] Estimating profit for crop='{crop}' area={area_hectares}ha price={price_per_ton}")
        try:
            return self._compute_profit(crop, area_hectares, price_per_ton)
        except Exception as e:
            logger.error(f"[PROFIT] Computation failed: {e}")
            return self._fallback_profit(crop, area_hectares, price_per_ton)

    def _compute_profit(self, crop: str, area: float, price: float) -> dict:
        yield_per_ha = self._get_yield(crop)
        cost_per_ha = DEFAULT_COST_PER_HECTARE.get(crop.lower(), 20000)

        total_yield = round(yield_per_ha * area, 2)
        total_revenue = round(total_yield * price, 2)
        total_cost = round(cost_per_ha * area, 2)
        net_profit = round(total_revenue - total_cost, 2)
        roi = round((net_profit / total_cost * 100) if total_cost > 0 else 0, 2)
        break_even_price = round(total_cost / total_yield if total_yield > 0 else 0, 2)

        return {
            "crop": crop,
            "area_hectares": area,
            "yield_tons": total_yield,
            "yield_per_hectare": yield_per_ha,
            "revenue": total_revenue,
            "cost": total_cost,
            "net_profit": net_profit,
            "roi_percent": roi,
            "break_even_price": break_even_price,
            "profitable": net_profit > 0,
            "status": "success",
        }

    def _get_yield(self, crop: str) -> float:
        crop_key = crop.lower()
        if not self.production_df.empty and "crop" in self.production_df.columns:
            df = self.production_df
            crop_df = df[df["crop"].str.lower() == crop_key]
            if not crop_df.empty and "area" in crop_df.columns and "production" in crop_df.columns:
                area_sum = crop_df["area"].sum()
                prod_sum = crop_df["production"].sum()
                if area_sum > 0:
                    return round(float(prod_sum / area_sum), 3)
        return DEFAULT_YIELD_PER_HECTARE.get(crop_key, 2.5)

    def _fallback_profit(self, crop: str, area: float, price: float) -> dict:
        logger.warning("[PROFIT] Using fallback profit estimation")
        yield_ph = DEFAULT_YIELD_PER_HECTARE.get(crop.lower(), 2.5)
        cost_ph = DEFAULT_COST_PER_HECTARE.get(crop.lower(), 20000)
        revenue = round(yield_ph * area * price, 2)
        cost = round(cost_ph * area, 2)
        return {
            "crop": crop, "area_hectares": area,
            "yield_tons": round(yield_ph * area, 2),
            "yield_per_hectare": yield_ph,
            "revenue": revenue, "cost": cost,
            "net_profit": round(revenue - cost, 2),
            "roi_percent": round((revenue - cost) / cost * 100 if cost > 0 else 0, 2),
            "break_even_price": round(cost / (yield_ph * area) if yield_ph * area > 0 else 0, 2),
            "profitable": revenue > cost,
            "status": "fallback used",
        }


# ─────────────────────────────────────────────
# TIMELINE GENERATOR
# ─────────────────────────────────────────────

CROP_TIMELINES = {
    "rice": [
        {"stage": "Land Preparation", "week": "1-2", "activity": "Plowing, leveling, flooding field"},
        {"stage": "Nursery/Sowing",   "week": "3-4", "activity": "Sow seeds in nursery beds"},
        {"stage": "Transplanting",    "week": "5-6", "activity": "Transplant seedlings to main field"},
        {"stage": "Vegetative Growth","week": "7-12","activity": "Fertilize, weed control, water management"},
        {"stage": "Panicle Initiation","week":"13-15","activity": "Apply potassium, monitor pests"},
        {"stage": "Flowering",        "week": "16-17","activity": "Critical water period, pest watch"},
        {"stage": "Grain Filling",    "week": "18-21","activity": "Reduce water, apply foliar feed"},
        {"stage": "Harvesting",       "week": "22-24","activity": "Harvest when 80% grains turn golden"},
        {"stage": "Post-Harvest",     "week": "25",   "activity": "Threshing, drying, storage"},
    ],
    "wheat": [
        {"stage": "Land Preparation", "week": "1",    "activity": "Deep plowing, add basal fertilizers"},
        {"stage": "Sowing",           "week": "2",    "activity": "Sow seeds 5-6 cm deep, row spacing 20cm"},
        {"stage": "Germination",      "week": "3-4",  "activity": "Ensure adequate moisture"},
        {"stage": "Tillering",        "week": "5-7",  "activity": "First irrigation, urea top-dressing"},
        {"stage": "Jointing",         "week": "8-10", "activity": "Second irrigation, weed removal"},
        {"stage": "Booting/Heading",  "week": "11-13","activity": "Third irrigation, fungicide if needed"},
        {"stage": "Grain Filling",    "week": "14-16","activity": "Fourth irrigation, monitor rust"},
        {"stage": "Maturity",         "week": "17-18","activity": "Withhold water, watch for shatter"},
        {"stage": "Harvesting",       "week": "19-20","activity": "Combine harvest at 14% moisture"},
    ],
    "maize": [
        {"stage": "Land Preparation", "week": "1",    "activity": "Till soil, apply compost"},
        {"stage": "Planting",         "week": "2",    "activity": "Plant seeds 4-5 cm deep"},
        {"stage": "Germination",      "week": "3",    "activity": "Ensure soil moisture"},
        {"stage": "Vegetative V6",    "week": "4-5",  "activity": "Side-dress nitrogen fertilizer"},
        {"stage": "Tasseling",        "week": "8-10", "activity": "Critical water stage, pest scouting"},
        {"stage": "Silking/Pollination","week":"10-11","activity": "Do not stress water, avoid pesticide"},
        {"stage": "Dough Stage",      "week": "12-14","activity": "Reduce irrigation, monitor earworm"},
        {"stage": "Maturity",         "week": "16-18","activity": "Black layer forms at kernel tip"},
        {"stage": "Harvesting",       "week": "19-20","activity": "Harvest at 25% moisture or dry in field"},
    ],
    "cotton": [
        {"stage": "Land Prep",        "week": "1-2",  "activity": "Deep plow, apply FYM 10t/ha"},
        {"stage": "Sowing",           "week": "3",    "activity": "Plant Bt cotton seeds with spacing 90x60cm"},
        {"stage": "Germination",      "week": "4",    "activity": "Gap filling, ensure stand"},
        {"stage": "Squaring",         "week": "7-9",  "activity": "Apply fertilizer, bollworm scouting"},
        {"stage": "Flowering",        "week": "10-13","activity": "Pest management critical phase"},
        {"stage": "Boll Development", "week": "14-18","activity": "Monitor pink bollworm, reduce water"},
        {"stage": "Boll Opening",     "week": "19-22","activity": "Defoliate, prepare for harvest"},
        {"stage": "Harvesting",       "week": "23-28","activity": "Manual or mechanical harvest 3-4 pickings"},
    ],
}

GENERIC_TIMELINE = [
    {"stage": "Land Preparation", "week": "1-2",  "activity": "Prepare soil, test and amend pH"},
    {"stage": "Sowing/Planting",  "week": "3",    "activity": "Plant seeds/seedlings at recommended spacing"},
    {"stage": "Early Growth",     "week": "4-6",  "activity": "Water regularly, apply basal fertilizer"},
    {"stage": "Vegetative Stage", "week": "7-10", "activity": "Weed control, pest scouting, irrigation"},
    {"stage": "Flowering",        "week": "11-14","activity": "Top-dress fertilizer, pollinator care"},
    {"stage": "Fruiting/Filling", "week": "15-18","activity": "Reduce water, disease monitoring"},
    {"stage": "Maturity",         "week": "19-20","activity": "Pre-harvest preparation"},
    {"stage": "Harvesting",       "week": "21-22","activity": "Harvest at optimal maturity, process and store"},
]


class TimelineGenerator:
    def generate(self, crop: str, season: str = "Kharif") -> dict:
        logger.info(f"[TIMELINE] Generating timeline for crop='{crop}' season='{season}'")
        try:
            return self._build_timeline(crop, season)
        except Exception as e:
            logger.error(f"[TIMELINE] Failed: {e}")
            return {"timeline": GENERIC_TIMELINE, "crop": crop, "season": season, "status": "fallback used"}

    def _build_timeline(self, crop: str, season: str) -> dict:
        crop_key = crop.lower()
        timeline = CROP_TIMELINES.get(crop_key, GENERIC_TIMELINE)
        is_generic = crop_key not in CROP_TIMELINES

        duration_weeks = int(timeline[-1]["week"].split("-")[0]) + 2 if "-" in timeline[-1]["week"] else int(timeline[-1]["week"]) + 2

        return {
            "crop": crop,
            "season": season,
            "duration_weeks": duration_weeks,
            "timeline": timeline,
            "is_generic": is_generic,
            "status": "success" if not is_generic else "generic timeline used",
        }
