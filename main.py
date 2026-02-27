from fastapi import FastAPI

app = FastAPI(
    title="Lieferkette Optimierungsplattform API",
    version="0.1.0",
    description="Supply Chain Optimization & Simulation API",
)

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