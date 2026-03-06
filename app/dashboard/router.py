from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <html>
        <head>
            <title>Supply Chain Dashboard</title>
        </head>
        <body>
            <h1>Supply Chain Optimization Dashboard</h1>
            <p>API is running successfully.</p>
            <p>Use /docs to test the API.</p>
        </body>
    </html>
    """