/**
 * FarmBrain Backend — Express Server
 * Self-healing, never crashes, structured error responses.
 */
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const logger = require('./utils/logger');
const apiRoutes = require('./routes/api');

const app = express();
const PORT = process.env.PORT || 3001;

// ─── Middleware ───────────────────────────────────────────
app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan('combined', {
  stream: { write: (msg) => logger.debug('HTTP', msg.trim()) }
}));

// Request logger
app.use((req, res, next) => {
  logger.info('SERVER', `${req.method} ${req.path}`);
  next();
});

// ─── Routes ──────────────────────────────────────────────
app.use('/api', apiRoutes);

app.get('/', (req, res) => {
  res.json({
    name: 'FarmBrain Backend',
    version: '1.0.0',
    status: 'running',
    endpoints: [
      'POST /api/predict',
      'POST /api/predict-price',
      'GET  /api/health',
      'GET  /api/crops',
    ],
  });
});

// ─── 404 Handler ─────────────────────────────────────────
app.use((req, res) => {
  logger.warn('SERVER', `404 — ${req.method} ${req.path}`);
  res.status(404).json({ status: 'error', message: `Route not found: ${req.method} ${req.path}` });
});

// ─── Global Error Handler ────────────────────────────────
app.use((err, req, res, _next) => {
  logger.error('SERVER', `Unhandled error: ${err.message}`, { stack: err.stack });
  res.status(500).json({
    status: 'error',
    message: 'Internal server error',
    detail: err.message,
  });
});

// ─── Uncaught Exception Guards ───────────────────────────
process.on('uncaughtException', (err) => {
  logger.error('PROCESS', `Uncaught Exception: ${err.message}`, { stack: err.stack });
  // Log but DO NOT exit — keep server alive
});

process.on('unhandledRejection', (reason) => {
  logger.error('PROCESS', `Unhandled Rejection: ${reason}`);
});

// ─── Start ───────────────────────────────────────────────
app.listen(PORT, () => {
  logger.info('SERVER', `🌾 FarmBrain Backend running on port ${PORT}`);
  logger.info('SERVER', `ML Service URL: ${process.env.ML_SERVICE_URL || 'http://localhost:8000'}`);
});

module.exports = app;
