"""
FarmBrain ML Service — FastAPI
Loads all models on startup, exposes prediction endpoints.
Never crashes — all errors produce safe fallback responses.
"""
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("farmbrain.main")

# Import engines
sys.path.insert(0, os.path.dirname(__file__))
from utils.data_validator import load_and_validate
from models.crop_engine import CropRecommendationEngine
from models.price_engine import PricePredictionEngine
from models.analytics import RiskAnalysisEngine, ProfitEstimator, TimelineGenerator

# Data path
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

# Global engine instances
crop_engine = CropRecommendationEngine()
price_engine = PricePredictionEngine()
risk_engine = RiskAnalysisEngine()
profit_estimator = ProfitEstimator()
timeline_gen = TimelineGenerator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load data and train models."""
    logger.info("=" * 60)
    logger.info("🌱 FarmBrain ML Service starting up...")
    logger.info("=" * 60)

    try:
        # Load datasets
        crop_df = load_and_validate(os.path.join(DATA_DIR, "crop_recommendation.csv"), "crop_recommendation")
        price_df = load_and_validate(os.path.join(DATA_DIR, "market_prices.csv"), "market_prices")
        production_df = load_and_validate(os.path.join(DATA_DIR, "crop_production.csv"), "crop_production")
        weather_df = load_and_validate(os.path.join(DATA_DIR, "weather_data.csv"), "weather_data")

        # Train/load all engines
        logger.info("[STARTUP] Training crop recommendation model...")
        crop_engine.train(crop_df)

        logger.info("[STARTUP] Loading price data...")
        price_engine.load_data(price_df)

        logger.info("[STARTUP] Loading weather data for risk analysis...")
        risk_engine.load_weather(weather_df)

        logger.info("[STARTUP] Loading production data for profit estimation...")
        profit_estimator.load_production(production_df)

        logger.info("✅ All systems initialized successfully!")

    except Exception as e:
        logger.error(f"[STARTUP ERROR] {e}")
        logger.error(traceback.format_exc())
        logger.warning("⚠️  System starting in degraded mode — fallbacks active")

    yield
    logger.info("🛑 FarmBrain ML Service shutting down")


app = FastAPI(
    title="FarmBrain ML Service",
    description="Intelligent farming decision platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────

class CropPredictRequest(BaseModel):
    N: float = Field(default=90.0, ge=0, le=200, description="Nitrogen content")
    P: float = Field(default=42.0, ge=0, le=200, description="Phosphorus content")
    K: float = Field(default=43.0, ge=0, le=200, description="Potassium content")
    temperature: float = Field(default=25.0, ge=-10, le=60, description="Temperature °C")
    humidity: float = Field(default=70.0, ge=0, le=100, description="Humidity %")
    ph: float = Field(default=6.5, ge=0, le=14, description="Soil pH")
    rainfall: float = Field(default=150.0, ge=0, le=500, description="Rainfall mm")
    area_hectares: float = Field(default=5.0, ge=0.1, le=10000)
    season: Optional[str] = "Kharif"


class PricePredictRequest(BaseModel):
    crop: str = Field(..., min_length=2, max_length=50)
    months_ahead: int = Field(default=3, ge=1, le=24)


# ─────────────────────────────────────────────
# Global Exception Handler
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"[UNHANDLED] {request.url} — {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "status": "error",
            "best_crop": "rice",
            "top_crops": ["rice", "wheat", "maize"],
            "confidence": [60.0, 25.0, 15.0],
            "profit": "N/A",
            "risk": "Medium",
            "timeline": [],
        },
    )


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "crop_model": crop_engine.mode,
        "price_model": "ready" if price_engine.is_ready else "fallback",
        "version": "1.0.0",
    }


# ─────────────────────────────────────────────
# /predict-crop
# ─────────────────────────────────────────────

@app.post("/predict-crop")
async def predict_crop(req: CropPredictRequest):
    logger.info(f"[API] /predict-crop called with N={req.N} P={req.P} K={req.K} temp={req.temperature}")
    try:
        # 1. Crop recommendation
        crop_result = crop_engine.predict(req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall)
        best_crop = crop_result["best_crop"]

        # 2. Price prediction for best crop
        price_result = price_engine.predict(best_crop, months_ahead=3)
        current_price = price_result["current_price"]

        # 3. Risk analysis
        volatility = price_engine.calculate_price_volatility(best_crop)
        risk_result = risk_engine.analyze(best_crop, req.rainfall, volatility)

        # 4. Profit estimation
        profit_result = profit_estimator.estimate(best_crop, req.area_hectares, current_price)

        # 5. Timeline
        timeline_result = timeline_gen.generate(best_crop, req.season or "Kharif")

        # Determine overall status
        statuses = [crop_result.get("status"), price_result.get("status"),
                    risk_result.get("status"), profit_result.get("status")]
        overall_status = "success" if all(s == "success" for s in statuses) else "partial-fallback"

        return {
            "best_crop": best_crop,
            "top_crops": crop_result["top_crops"],
            "confidence": crop_result["confidence"],
            "model_mode": crop_result["mode"],
            "profit": f"₹{profit_result['net_profit']:,.0f}",
            "profit_details": profit_result,
            "risk": risk_result["risk_level"],
            "risk_details": risk_result,
            "price_forecast": price_result,
            "timeline": timeline_result["timeline"],
            "timeline_details": timeline_result,
            "status": overall_status,
        }

    except Exception as e:
        logger.error(f"[API ERROR] /predict-crop: {e}\n{traceback.format_exc()}")
        return {
            "best_crop": "rice",
            "top_crops": ["rice", "wheat", "maize"],
            "confidence": [60.0, 25.0, 15.0],
            "profit": "₹45,000",
            "profit_details": {},
            "risk": "Medium",
            "risk_details": {"risk_factors": ["System in fallback mode"]},
            "price_forecast": {"current_price": 2000, "price_trend": "stable"},
            "timeline": timeline_gen.generate("rice", "Kharif")["timeline"],
            "timeline_details": {},
            "status": "fallback used",
        }


# ─────────────────────────────────────────────
# /predict-price
# ─────────────────────────────────────────────

@app.post("/predict-price")
async def predict_price(req: PricePredictRequest):
    logger.info(f"[API] /predict-price crop={req.crop} months={req.months_ahead}")
    try:
        result = price_engine.predict(req.crop, req.months_ahead)
        return result
    except Exception as e:
        logger.error(f"[API ERROR] /predict-price: {e}")
        return {
            "crop": req.crop,
            "current_price": 2000.0,
            "predicted_prices": [2060.0, 2122.0, 2186.0][:req.months_ahead],
            "price_trend": "stable",
            "change_percent": 3.0,
            "mode": "error-fallback",
            "status": "fallback used",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
