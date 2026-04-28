# 🌾 FarmBrain – End-to-End Farming Decision System

**Tagline:** *From soil to selling — AI decides everything*

---

## 🚨 The Problem

Farmers face **5 critical decisions** every season — often made blindly:

1. ❓ What crop to grow  
2. ❓ When to sow  
3. ❓ How much to invest  
4. ❓ When to harvest  
5. ❓ When & where to sell  

These decisions affect income, risk, and sustainability — yet most farmers lack data-driven guidance.

---

## 💡 The Solution

**FarmBrain** is a web platform that provides **complete, AI-powered farming decisions** in one place.  
No more guesswork. No more scattered advice. Just **clear, actionable answers** from soil to selling.

---

## 🧠 How It Works (Intelligence Layer)

FarmBrain combines multiple data sources and models:

| Feature | Intelligence Used |
|--------|-------------------|
| **Crop recommendation** | Soil nutrients + historical weather patterns |
| **Price prediction** | Market trends + seasonal demand |
| **Risk score** | Rainfall forecasts, pest cycles, temperature volatility |
| **Profit estimation** | Input costs + predicted yield + selling price |

---

## 🌐 Features

### User Inputs
- 📍 Location  
- 🌾 Land size (acres/hectares)  
- 💰 Budget (for seeds, fertilizer, labor, etc.)

### System Outputs
- ✅ **Best crop to grow**  
- 💵 **Expected profit**  
- ⚠️ **Risk level** (Low / Medium / High)  
- 📅 **Timeline** – From sowing → harvest → selling window

---

## 🖥️ Tech Stack (Suggested)

| Layer | Technology |
|-------|------------|
| Frontend | React / Next.js |
| Backend | Node.js / FastAPI (Python) |
| ML Models | Scikit-learn, XGBoost, LSTM (price prediction) |
| Data Sources | OpenWeather, SoilGrids, Govt. market price APIs |
| Database | PostgreSQL |
| Deployment | AWS / Vercel / Render |

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.9+)
- API keys for weather & market data

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/farmbrain.git
cd farmbrain

# Backend setup
cd backend
pip install -r requirements.txt
python app.py

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
