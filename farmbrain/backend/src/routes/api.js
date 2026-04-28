/**
 * FarmBrain API Routes
 * Proxies to ML service with full error handling and fallbacks.
 */
const express = require('express');
const axios = require('axios');
const { cropPredictRules, pricePredictRules, validate } = require('../middleware/validator');
const logger = require('../utils/logger');

const router = express.Router();
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

const ML_TIMEOUT = 30000; // 30s

// Static fallback response for when ML service is unreachable
const FALLBACK_PREDICTION = {
  best_crop: 'rice',
  top_crops: ['rice', 'wheat', 'maize'],
  confidence: [60.0, 25.0, 15.0],
  profit: '₹45,000',
  profit_details: { note: 'Estimated from averages' },
  risk: 'Medium',
  risk_details: { risk_factors: ['ML service unavailable — using static fallback'] },
  price_forecast: { current_price: 2000, price_trend: 'stable', predicted_prices: [2060, 2122, 2186] },
  timeline: [
    { stage: 'Land Preparation', week: '1-2',  activity: 'Prepare soil, test pH' },
    { stage: 'Sowing',           week: '3',    activity: 'Plant seeds at recommended spacing' },
    { stage: 'Growth',           week: '4-10', activity: 'Water, fertilize, weed control' },
    { stage: 'Harvesting',       week: '18-22','activity: Harvest at optimal maturity' },
  ],
  timeline_details: {},
  status: 'fallback used — ML service unavailable',
};

/**
 * POST /api/predict
 * Full crop recommendation + all analytics
 */
router.post('/predict', cropPredictRules, validate, async (req, res) => {
  logger.info('ROUTES', 'POST /api/predict', { body: req.body });

  const payload = {
    N: Number(req.body.N ?? 90),
    P: Number(req.body.P ?? 42),
    K: Number(req.body.K ?? 43),
    temperature: Number(req.body.temperature ?? 25),
    humidity: Number(req.body.humidity ?? 70),
    ph: Number(req.body.ph ?? 6.5),
    rainfall: Number(req.body.rainfall ?? 150),
    area_hectares: Number(req.body.area_hectares ?? 5),
    season: req.body.season || 'Kharif',
  };

  try {
    logger.info('ROUTES', 'Calling ML service /predict-crop', payload);
    const mlRes = await axios.post(`${ML_SERVICE_URL}/predict-crop`, payload, { timeout: ML_TIMEOUT });
    const data = mlRes.data;

    logger.info('ROUTES', 'ML service responded', { status: data.status, crop: data.best_crop });
    return res.json({ ...data, _source: 'ml-service' });

  } catch (err) {
    logger.error('ROUTES', `ML service call failed: ${err.message}`);

    if (err.code === 'ECONNREFUSED' || err.code === 'ECONNRESET' || err.code === 'ETIMEDOUT') {
      logger.warn('ROUTES', 'ML service unreachable — serving static fallback');
      return res.json({ ...FALLBACK_PREDICTION, _source: 'static-fallback' });
    }

    if (err.response && err.response.status === 422) {
      return res.status(422).json({
        status: 'validation_error',
        errors: err.response.data,
        _source: 'ml-service-validation',
      });
    }

    logger.warn('ROUTES', 'Unknown error — serving static fallback');
    return res.json({ ...FALLBACK_PREDICTION, _source: 'error-fallback', _error: err.message });
  }
});

/**
 * POST /api/predict-price
 */
router.post('/predict-price', pricePredictRules, validate, async (req, res) => {
  logger.info('ROUTES', 'POST /api/predict-price', { crop: req.body.crop });

  try {
    const mlRes = await axios.post(`${ML_SERVICE_URL}/predict-price`, {
      crop: req.body.crop,
      months_ahead: Number(req.body.months_ahead ?? 3),
    }, { timeout: ML_TIMEOUT });
    return res.json(mlRes.data);
  } catch (err) {
    logger.error('ROUTES', `Price predict failed: ${err.message}`);
    return res.json({
      crop: req.body.crop,
      current_price: 2000,
      predicted_prices: [2060, 2122, 2186],
      price_trend: 'stable',
      change_percent: 3.0,
      status: 'fallback used',
      _source: 'static-fallback',
    });
  }
});

/**
 * GET /api/health
 */
router.get('/health', async (req, res) => {
  let mlHealth = { status: 'unreachable' };
  try {
    const r = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 5000 });
    mlHealth = r.data;
  } catch (e) {
    logger.warn('ROUTES', 'ML health check failed');
  }
  res.json({
    backend: 'ok',
    ml_service: mlHealth,
    timestamp: new Date().toISOString(),
  });
});

/**
 * GET /api/crops — list of supported crops
 */
router.get('/crops', (req, res) => {
  res.json({
    crops: ['rice','wheat','maize','chickpea','kidneybeans','pigeonpeas','mothbeans','mungbean',
            'blackgram','lentil','pomegranate','banana','mango','grapes','watermelon','muskmelon',
            'apple','orange','papaya','coconut','cotton','jute','coffee'],
    seasons: ['Kharif','Rabi','Whole Year','Summer','Winter','Autumn'],
  });
});

module.exports = router;
