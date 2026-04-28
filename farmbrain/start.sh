#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# FarmBrain — Startup Script
# Starts ML service, backend, and opens frontend
# ═══════════════════════════════════════════════════════════

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "🌾 ╔══════════════════════════════════════╗"
echo "   ║    FarmBrain Intelligent Platform    ║"
echo "   ╚══════════════════════════════════════╝"
echo ""

# ─── Check dependencies ───────────────────────────────────
echo "📦 Checking dependencies..."
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ node required"; exit 1; }
python3 -c "import fastapi, uvicorn, sklearn, pandas, numpy" 2>/dev/null || {
  echo "⚠️  Installing Python deps..."
  pip install fastapi uvicorn pandas scikit-learn numpy statsmodels python-multipart --break-system-packages -q
}

# ─── Generate sample data if needed ──────────────────────
if [ ! -f "$ROOT/data/crop_recommendation.csv" ]; then
  echo "📊 Generating sample datasets..."
  cd "$ROOT/data" && python3 generate_sample_data.py
fi

# ─── Start ML Service ─────────────────────────────────────
echo ""
echo "🧠 Starting ML Service (port 8000)..."
export DATA_DIR="$ROOT/data"
cd "$ROOT/ml-service"
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 &
ML_PID=$!
echo "   ML PID: $ML_PID"

# ─── Wait for ML service ──────────────────────────────────
echo "   Waiting for ML service to initialize..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "   ✅ ML Service ready!"
    break
  fi
  sleep 1
done

# ─── Start Backend ─────────────────────────────────────────
echo ""
echo "🌐 Starting Backend (port 3001)..."
cd "$ROOT/backend"
[ ! -d node_modules ] && npm install -q
ML_SERVICE_URL=http://localhost:8000 node src/index.js &
BE_PID=$!
echo "   Backend PID: $BE_PID"
sleep 2

# ─── Run Tests ─────────────────────────────────────────────
echo ""
echo "🧪 Running test suite..."
cd "$ROOT"
python3 tests/test_suite.py 2>/dev/null
echo ""

# ─── Done ──────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo "✅ FarmBrain is running!"
echo ""
echo "   🌐 Frontend:   open frontend/farmbrain.html in browser"
echo "   🧠 ML Service: http://localhost:8000"
echo "   🔌 Backend:    http://localhost:3001"
echo "   📋 API Docs:   http://localhost:8000/docs"
echo "   ❤️  Health:     http://localhost:3001/api/health"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop all services"

# ─── Cleanup on exit ──────────────────────────────────────
trap "echo ''; echo '🛑 Stopping services...'; kill $ML_PID $BE_PID 2>/dev/null; exit 0" INT TERM

wait
