from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routers.optimize import router as optimize_router
from app.routers.simulation_router import router as simulation_router
from app.dashboard.router import router as dashboard_router


app = FastAPI(
    title="Lieferkette Optimierungsplattform API",
    version="0.2.0"
)


# CORS (اجازه دسترسی برای Dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(optimize_router)
app.include_router(simulation_router)
app.include_router(dashboard_router)


# Serve Dashboard HTML
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")


@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}


@app.get("/health")
def health():
    return {"ok": True}