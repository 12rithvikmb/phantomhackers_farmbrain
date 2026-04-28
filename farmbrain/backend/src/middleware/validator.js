/**
 * FarmBrain Input Validation Middleware
 * Validates all API inputs strictly, returns structured errors.
 */
const { body, validationResult } = require('express-validator');

const cropPredictRules = [
  body('N').optional().isFloat({ min: 0, max: 200 }).withMessage('N must be 0–200'),
  body('P').optional().isFloat({ min: 0, max: 200 }).withMessage('P must be 0–200'),
  body('K').optional().isFloat({ min: 0, max: 200 }).withMessage('K must be 0–200'),
  body('temperature').optional().isFloat({ min: -10, max: 60 }).withMessage('Temperature must be -10 to 60°C'),
  body('humidity').optional().isFloat({ min: 0, max: 100 }).withMessage('Humidity must be 0–100%'),
  body('ph').optional().isFloat({ min: 0, max: 14 }).withMessage('pH must be 0–14'),
  body('rainfall').optional().isFloat({ min: 0, max: 500 }).withMessage('Rainfall must be 0–500mm'),
  body('area_hectares').optional().isFloat({ min: 0.1, max: 10000 }).withMessage('Area must be 0.1–10000 hectares'),
  body('season').optional().isString().isIn(['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter', 'Autumn', 'Zaid'])
    .withMessage('Invalid season'),
];

const pricePredictRules = [
  body('crop').notEmpty().isString().isLength({ min: 2, max: 50 }).withMessage('Crop name required (2–50 chars)'),
  body('months_ahead').optional().isInt({ min: 1, max: 24 }).withMessage('months_ahead must be 1–24'),
];

function validate(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(422).json({
      status: 'validation_error',
      errors: errors.array().map(e => ({ field: e.path, message: e.msg, value: e.value })),
    });
  }
  next();
}

module.exports = { cropPredictRules, pricePredictRules, validate };
