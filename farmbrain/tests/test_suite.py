"""
FarmBrain Test Suite
Tests all engines including edge cases: missing data, invalid inputs, model failures.
"""
import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml-service'))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.WARNING)  # quiet during tests

from utils.data_validator import load_and_validate
from models.crop_engine import CropRecommendationEngine
from models.price_engine import PricePredictionEngine
from models.analytics import RiskAnalysisEngine, ProfitEstimator, TimelineGenerator

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def test(name, fn):
    try:
        fn()
        results.append((name, True, None))
        print(f"  {PASS} — {name}")
    except AssertionError as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} — {name}: {e}")
    except Exception as e:
        results.append((name, False, f"Exception: {e}"))
        print(f"  {FAIL} — {name}: Exception: {e}")


# ─── DATA VALIDATOR TESTS ────────────────────────────────────

print("\n📋 DATA VALIDATOR TESTS")

def t_missing_file():
    df = load_and_validate("/nonexistent/path/file.csv", "crop_recommendation")
    assert not df.empty, "Should return fallback data for missing file"
test("Missing file returns fallback data", t_missing_file)

def t_valid_file():
    df = load_and_validate("data/crop_recommendation.csv", "crop_recommendation")
    assert len(df) > 100, "Should load > 100 rows"
    assert "label" in df.columns, "Should have label column"
test("Valid file loads correctly", t_valid_file)

def t_normalize_columns():
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["  N ", "  P  ", "Label"])
        writer.writerow([90, 42, "rice"])
        fname = f.name
    df = load_and_validate(fname, "crop_recommendation")
    assert "n" in df.columns, "Column names should be normalized"
    os.unlink(fname)
test("Column names normalized", t_normalize_columns)

def t_missing_value_imputation():
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["N","P","K","temperature","humidity","ph","rainfall","label"])
        writer.writerow([90, "", 43, 25.0, 70.0, 6.5, 150.0, "rice"])
        writer.writerow(["", 42, 43, 25.0, 70.0, 6.5, 150.0, "wheat"])
        fname = f.name
    df = load_and_validate(fname, "crop_recommendation")
    assert not df["n"].isna().any(), "Numeric nulls should be imputed"
    assert not df["p"].isna().any(), "Numeric nulls should be imputed"
    os.unlink(fname)
test("Missing values imputed", t_missing_value_imputation)

def t_duplicate_removal():
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["N","P","K","temperature","humidity","ph","rainfall","label"])
        for _ in range(5): writer.writerow([90, 42, 43, 25.0, 70.0, 6.5, 150.0, "rice"])
        fname = f.name
    df = load_and_validate(fname, "crop_recommendation")
    assert len(df) == 1, f"Should remove duplicates, got {len(df)} rows"
    os.unlink(fname)
test("Duplicate rows removed", t_duplicate_removal)


# ─── CROP ENGINE TESTS ───────────────────────────────────────

print("\n🌾 CROP ENGINE TESTS")

crop_df = load_and_validate("data/crop_recommendation.csv", "crop_recommendation")
ce = CropRecommendationEngine()
ce.train(crop_df)

def t_crop_predict_normal():
    r = ce.predict(90, 42, 43, 20.9, 82.0, 6.5, 202.9)
    assert "best_crop" in r
    assert "top_crops" in r
    assert len(r["top_crops"]) == 3
    assert "confidence" in r
    assert all(0 <= c <= 100 for c in r["confidence"])
test("Normal crop prediction returns 3 crops with confidence", t_crop_predict_normal)

def t_crop_predict_edge_zeros():
    r = ce.predict(0, 0, 0, 0, 0, 0, 0)
    assert "best_crop" in r and r["best_crop"]
test("Crop predict with all-zero inputs doesn't crash", t_crop_predict_edge_zeros)

def t_crop_predict_extreme_values():
    r = ce.predict(200, 200, 200, 60, 100, 14, 500)
    assert "best_crop" in r
test("Crop predict with extreme values doesn't crash", t_crop_predict_extreme_values)

def t_crop_rule_based_fallback():
    ce_tmp = CropRecommendationEngine()
    ce_tmp.mode = "rule-based"
    ce_tmp.is_trained = False
    r = ce_tmp.predict(90, 42, 43, 25.0, 70.0, 6.5, 200.0)
    assert r["mode"] == "rule-based"
    assert r["best_crop"]
test("Rule-based fallback works when ML fails", t_crop_rule_based_fallback)

def t_crop_train_empty_df():
    ce_tmp = CropRecommendationEngine()
    ce_tmp.train(pd.DataFrame())  # should not crash
    r = ce_tmp.predict(90, 42, 43, 25.0, 70.0, 6.5, 200.0)
    assert r["best_crop"]  # fallback should kick in
test("Training on empty dataframe falls back gracefully", t_crop_train_empty_df)

def t_crop_train_missing_label():
    df_no_label = crop_df.drop(columns=["label"])
    ce_tmp = CropRecommendationEngine()
    ce_tmp.train(df_no_label)
    r = ce_tmp.predict(90, 42, 43, 25.0, 70.0, 6.5, 200.0)
    assert "best_crop" in r
test("Training without label column falls back gracefully", t_crop_train_missing_label)


# ─── PRICE ENGINE TESTS ──────────────────────────────────────

print("\n💰 PRICE ENGINE TESTS")

price_df = load_and_validate("data/market_prices.csv", "market_prices")
pe = PricePredictionEngine()
pe.load_data(price_df)

def t_price_known_crop():
    r = pe.predict("rice", 3)
    assert "current_price" in r
    assert len(r["predicted_prices"]) == 3
    assert r["current_price"] > 0
test("Price prediction for known crop", t_price_known_crop)

def t_price_unknown_crop():
    r = pe.predict("unknowncrop123", 3)
    assert "current_price" in r
    assert r["status"] == "fallback used"
test("Price prediction for unknown crop uses fallback", t_price_unknown_crop)

def t_price_empty_df():
    pe_tmp = PricePredictionEngine()
    pe_tmp.load_data(pd.DataFrame())
    r = pe_tmp.predict("rice", 3)
    assert "current_price" in r
test("Price engine with empty dataframe returns fallback", t_price_empty_df)

def t_price_months_ahead():
    for months in [1, 6, 12, 24]:
        r = pe.predict("wheat", months)
        assert len(r["predicted_prices"]) == months, f"Expected {months} predictions"
test("Price prediction returns correct number of months", t_price_months_ahead)

def t_price_no_negative():
    r = pe.predict("maize", 12)
    assert all(p >= 0 for p in r["predicted_prices"]), "No negative prices allowed"
test("Predicted prices are never negative", t_price_no_negative)


# ─── RISK ENGINE TESTS ───────────────────────────────────────

print("\n⚠️  RISK ENGINE TESTS")

wx_df = load_and_validate("data/weather_data.csv", "weather_data")
re = RiskAnalysisEngine()
re.load_weather(wx_df)

def t_risk_normal():
    r = re.analyze("rice", 200.0, 15.0)
    assert r["risk_level"] in ["Low", "Medium", "High"]
    assert "risk_factors" in r
    assert "mitigation" in r
test("Risk analysis returns valid levels", t_risk_normal)

def t_risk_missing_data():
    re_empty = RiskAnalysisEngine()
    re_empty.load_weather(pd.DataFrame())
    r = re_empty.analyze("rice", 200.0, 15.0)
    assert r["risk_level"] == "Medium"  # safe default
test("Risk with missing weather data returns Medium default", t_risk_missing_data)

def t_risk_high_volatility():
    r = re.analyze("cotton", 50.0, 40.0)  # high price volatility
    assert r["risk_level"] in ["Medium", "High"]
test("High volatility crop gets elevated risk", t_risk_high_volatility)


# ─── PROFIT ESTIMATOR TESTS ──────────────────────────────────

print("\n📊 PROFIT ESTIMATOR TESTS")

prod_df = load_and_validate("data/crop_production.csv", "crop_production")
prof = ProfitEstimator()
prof.load_production(prod_df)

def t_profit_normal():
    r = prof.estimate("rice", 5.0, 2000.0)
    assert "net_profit" in r
    assert "yield_tons" in r
    assert r["yield_tons"] > 0
test("Profit estimate for known crop", t_profit_normal)

def t_profit_unknown_crop():
    r = prof.estimate("unknowncrop", 5.0, 2000.0)
    assert "net_profit" in r
    assert r["yield_per_hectare"] == 2.5  # fallback average
test("Profit estimate for unknown crop uses fallback yield", t_profit_unknown_crop)

def t_profit_tiny_area():
    r = prof.estimate("wheat", 0.1, 1500.0)
    assert r["yield_tons"] > 0
    assert "profitable" in r
test("Profit estimate for tiny area works", t_profit_tiny_area)

def t_profit_zero_price():
    r = prof.estimate("maize", 5.0, 0.0)
    assert r["revenue"] == 0.0
    assert r["profitable"] == False
test("Zero price results in zero revenue and not profitable", t_profit_zero_price)


# ─── TIMELINE TESTS ──────────────────────────────────────────

print("\n📅 TIMELINE TESTS")

tl = TimelineGenerator()

def t_timeline_known():
    r = tl.generate("rice", "Kharif")
    assert len(r["timeline"]) > 5
    assert r["status"] == "success"
test("Timeline for known crop returns full stages", t_timeline_known)

def t_timeline_unknown():
    r = tl.generate("xyz_alien_crop", "Kharif")
    assert len(r["timeline"]) > 0
    assert r["status"] == "generic timeline used"
test("Timeline for unknown crop returns generic template", t_timeline_unknown)

def t_timeline_all_known_crops():
    known = ["rice", "wheat", "maize", "cotton"]
    for crop in known:
        r = tl.generate(crop)
        assert r["timeline"], f"Empty timeline for {crop}"
test("All known crops have non-empty timelines", t_timeline_all_known_crops)


# ─── INTEGRATION TEST ────────────────────────────────────────

print("\n🔗 INTEGRATION TEST")

def t_full_pipeline():
    """Simulate full pipeline from raw input to structured output."""
    # Input
    N, P, K = 90, 42, 43
    temp, hum, ph, rain = 20.9, 82.0, 6.5, 202.9
    area = 5.0

    crop_result = ce.predict(N, P, K, temp, hum, ph, rain)
    best = crop_result["best_crop"]

    price_result = pe.predict(best, 3)
    risk_result = re.analyze(best, rain, pe.calculate_price_volatility(best))
    profit_result = prof.estimate(best, area, price_result["current_price"])
    tl_result = tl.generate(best, "Kharif")

    output = {
        "best_crop": best,
        "top_crops": crop_result["top_crops"],
        "confidence": crop_result["confidence"],
        "profit": f"₹{profit_result['net_profit']:,.0f}",
        "risk": risk_result["risk_level"],
        "timeline": tl_result["timeline"],
        "status": "success",
    }

    assert output["best_crop"]
    assert len(output["top_crops"]) == 3
    assert len(output["confidence"]) == 3
    assert output["profit"]
    assert output["risk"] in ["Low", "Medium", "High"]
    assert len(output["timeline"]) > 3

test("Full pipeline integration test", t_full_pipeline)


# ─── SUMMARY ─────────────────────────────────────────────────

total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n{'='*60}")
print(f"TEST RESULTS: {passed}/{total} passed")
if failed:
    print(f"\nFailed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  ❌ {name}: {err}")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)
