"""
FarmBrain Data Validation Layer
Handles missing files, bad columns, nulls, duplicates, and normalization.
"""
import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger("farmbrain.validator")

REQUIRED_SCHEMAS = {
    "crop_recommendation": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"],
    "crop_production": ["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"],
    "market_prices": ["crop", "month", "year", "min_price", "max_price", "modal_price"],
    "weather_data": ["date", "temperature", "humidity", "rainfall", "wind_speed", "location"],
}

DEFAULT_SAMPLES = {
    "crop_recommendation": pd.DataFrame([
        {"N": 90, "P": 42, "K": 43, "temperature": 20.88, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9, "label": "rice"},
        {"N": 85, "P": 58, "K": 41, "temperature": 21.77, "humidity": 80.3, "ph": 7.0, "rainfall": 226.6, "label": "maize"},
        {"N": 60, "P": 55, "K": 44, "temperature": 23.00, "humidity": 82.0, "ph": 7.8, "rainfall": 263.9, "label": "chickpea"},
        {"N": 74, "P": 35, "K": 40, "temperature": 26.49, "humidity": 80.2, "ph": 6.9, "rainfall": 150.3, "label": "wheat"},
        {"N": 20, "P": 30, "K": 20, "temperature": 25.00, "humidity": 70.0, "ph": 6.5, "rainfall": 100.0, "label": "cotton"},
    ]),
    "market_prices": pd.DataFrame([
        {"crop": "rice", "month": "Jan", "year": 2023, "min_price": 1800, "max_price": 2200, "modal_price": 2000},
        {"crop": "wheat", "month": "Jan", "year": 2023, "min_price": 1500, "max_price": 1900, "modal_price": 1700},
        {"crop": "maize", "month": "Jan", "year": 2023, "min_price": 1200, "max_price": 1600, "modal_price": 1400},
    ]),
    "crop_production": pd.DataFrame([
        {"State_Name": "Punjab", "District_Name": "Amritsar", "Crop_Year": 2022, "Season": "Rabi", "Crop": "wheat", "Area": 5000.0, "Production": 17500.0},
        {"State_Name": "UP", "District_Name": "Lucknow", "Crop_Year": 2022, "Season": "Kharif", "Crop": "rice", "Area": 4000.0, "Production": 14000.0},
    ]),
    "weather_data": pd.DataFrame([
        {"date": "2023-01-01", "temperature": 25.0, "humidity": 65.0, "rainfall": 5.0, "wind_speed": 12.0, "location": "North"},
        {"date": "2023-01-02", "temperature": 27.0, "humidity": 60.0, "rainfall": 0.0, "wind_speed": 10.0, "location": "North"},
    ]),
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_and_validate(filepath: str, dataset_name: str) -> pd.DataFrame:
    """
    Load CSV with full validation pipeline.
    Falls back to default sample data if file is missing or unreadable.
    """
    logger.info(f"[LOAD] Loading dataset '{dataset_name}' from {filepath}")

    # Step 1: File existence check
    if not os.path.exists(filepath):
        logger.error(f"[ERROR] File not found: {filepath}")
        logger.warning(f"[FALLBACK] Using default sample data for '{dataset_name}'")
        logger.warning(f"[FIX] Place your CSV at: {filepath}")
        return DEFAULT_SAMPLES.get(dataset_name, pd.DataFrame()).copy()

    # Step 2: Load CSV
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"[LOAD] Loaded {len(df)} rows, {len(df.columns)} cols from {dataset_name}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to read CSV {filepath}: {e}")
        logger.warning(f"[FALLBACK] Using default sample data for '{dataset_name}'")
        return DEFAULT_SAMPLES.get(dataset_name, pd.DataFrame()).copy()

    # Step 3: Normalize columns
    df = normalize_columns(df)
    logger.info(f"[CLEAN] Normalized column names: {list(df.columns)}")

    # Step 4: Check required columns
    required = [c.lower() for c in REQUIRED_SCHEMAS.get(dataset_name, [])]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        logger.warning(f"[WARN] Missing columns in {dataset_name}: {missing_cols}")

    # Step 5: Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info(f"[CLEAN] Removed {removed} duplicate rows from {dataset_name}")

    # Step 6: Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logger.info(f"[CLEAN] Imputed {null_count} nulls in '{col}' with median={fill_val:.2f}")

    for col in categorical_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(mode_val)
            logger.info(f"[CLEAN] Imputed {null_count} nulls in '{col}' with mode='{mode_val}'")

    logger.info(f"[DONE] Dataset '{dataset_name}' ready: {len(df)} rows, {len(df.columns)} cols")
    return df
