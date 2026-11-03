
import os
import httpx
import nest_asyncio
import uvicorn

from datetime import datetime
from typing import Optional, Literal

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# DATABASE
DATABASE_URL = "sqlite:///./aquaguard.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# DATABASE MODELS
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    hazard = Column(String(30))
    location = Column(String(150))
    score = Column(Float)
    risk_level = Column(String(30))
    created_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200))
    hazard = Column(String(30))
    location = Column(String(150))
    severity = Column(String(30))
    message = Column(Text)
    status = Column(String(30), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

class CommunityReport(Base):
    __tablename__ = "community_reports"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    location = Column(String(150))
    category = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# INPUT SCHEMAS
class RiskInput(BaseModel):
    location: str
    rainfall_mm_24h: float = Field(default=0, ge=0)
    forecast_rainfall_mm: float = Field(default=0, ge=0)
    soil_moisture_pct: float = Field(default=50, ge=0, le=100)
    river_level_pct: float = Field(default=30, ge=0, le=100)
    temperature_c: float = 28
    humidity_pct: float = Field(default=60, ge=0, le=100)
    rainfall_deficit_pct: float = Field(default=0, ge=0, le=100)

class AlertInput(BaseModel):
    title: str
    hazard: Literal["flood", "drought", "weather", "other"]
    location: str
    severity: Literal["low", "moderate", "high", "critical"]
    message: str

class ReportInput(BaseModel):
    name: str = "Anonymous"
    location: str
    category: str
    description: str

class AlertUpdate(BaseModel):
    status: Literal["active", "acknowledged", "resolved"]

# FASTAPI APP
app = FastAPI(
    title="AquaGuard AI",
    description="Flood and Drought Early Warning System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {
        "project": "AquaGuard AI",
        "status": "Backend Running",
        "docs": "/docs"
    }

@app.get("/api/health")
def health():
    return {"status": "healthy"}

# RISK CLASSIFICATION
def classify_risk(score):
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 35:
        return "Moderate"
    return "Low"

# FLOOD PREDICTION
@app.post("/api/predict/flood")
def predict_flood(data: RiskInput, db: Session = Depends(get_db)):

    rainfall = min(data.rainfall_mm_24h / 200, 1) * 35
    forecast = min(data.forecast_rainfall_mm / 200, 1) * 25
    river = data.river_level_pct * 0.25
    soil = data.soil_moisture_pct * 0.15

    score = round(min(100, rainfall + forecast + river + soil), 2)
    risk = classify_risk(score)

    record = Prediction(
        hazard="Flood",
        location=data.location,
        score=score,
        risk_level=risk
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "hazard": "Flood",
        "location": data.location,
        "risk_score": score,
        "risk_level": risk,
        "data_type": "DEMO_RULE_BASED_NOT_VALIDATED"
    }

# DROUGHT PREDICTION
@app.post("/api/predict/drought")
def predict_drought(data: RiskInput, db: Session = Depends(get_db)):

    deficit = data.rainfall_deficit_pct * 0.40
    soil = (100 - data.soil_moisture_pct) * 0.35
    heat = min(max(data.temperature_c - 20, 0), 25) / 25 * 15
    humidity = (100 - data.humidity_pct) * 0.10

    score = round(min(100, deficit + soil + heat + humidity), 2)
    risk = classify_risk(score)

    record = Prediction(
        hazard="Drought",
        location=data.location,
        score=score,
        risk_level=risk
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "hazard": "Drought",
        "location": data.location,
        "risk_score": score,
        "risk_level": risk,
        "data_type": "DEMO_RULE_BASED_NOT_VALIDATED"
    }

# PREDICTION HISTORY
@app.get("/api/predictions")
def get_predictions(db: Session = Depends(get_db)):
    return db.query(Prediction).order_by(
        Prediction.created_at.desc()
    ).limit(200).all()

# WEATHER API
@app.get("/api/weather/current")
async def current_weather(latitude: float, longitude: float):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "auto"
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params=params
            )
            response.raise_for_status()

        return {
            "source": "Open-Meteo",
            "weather": response.json()
        }

    except httpx.HTTPError:
        raise HTTPException(502, "Weather service unavailable")

# CREATE ALERT
@app.post("/api/alerts")
def create_alert(data: AlertInput, db: Session = Depends(get_db)):
    alert = Alert(**data.model_dump())
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert

# GET ALERTS
@app.get("/api/alerts")
def get_alerts(status: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Alert)

    if status:
        query = query.filter(Alert.status == status)

    return query.order_by(Alert.created_at.desc()).limit(200).all()

# UPDATE ALERT
@app.patch("/api/alerts/{alert_id}")
def update_alert(alert_id: int, data: AlertUpdate, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()

    if not alert:
        raise HTTPException(404, "Alert not found")

    alert.status = data.status
    db.commit()
    db.refresh(alert)
    return alert

# COMMUNITY REPORT
@app.post("/api/reports")
def create_report(data: ReportInput, db: Session = Depends(get_db)):
    report = CommunityReport(**data.model_dump())
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

@app.get("/api/reports")
def get_reports(db: Session = Depends(get_db)):
    return db.query(CommunityReport).order_by(
        CommunityReport.created_at.desc()
    ).limit(200).all()

# DASHBOARD SUMMARY
@app.get("/api/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    return {
        "active_alerts": db.query(Alert).filter(
            Alert.status == "active"
        ).count(),

        "high_risk_predictions": db.query(Prediction).filter(
            Prediction.risk_level.in_(["High", "Critical"])
        ).count(),

        "total_reports": db.query(CommunityReport).count(),

        "total_predictions": db.query(Prediction).count()
    }
