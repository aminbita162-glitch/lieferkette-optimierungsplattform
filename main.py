from fastapi import FastAPI
from app.routers.optimize import router as optimize_router

app = FastAPI(
    title="Lieferkette Optimierungsplattform API",
    version="0.2.0",
    description="Supply Chain Optimization & Simulation API",
)

app.include_router(optimize_router)

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "API is running",
        "docs": "/docs",
        "redoc": "/redoc",
    }

@app.get("/health")
def health():
    return {"ok": True}