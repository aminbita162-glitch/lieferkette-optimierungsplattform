from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from app.routers.optimize import router as optimize_router
from app.routers.simulation_router import router as simulation_router
from app.routers.ai_router import router as ai_router


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
app = FastAPI(
    title="Lieferkette Optimierungsplattform API",
    version="0.3.0"
)


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Existing API routers
# -------------------------------------------------------------------
app.include_router(optimize_router)
app.include_router(simulation_router)
app.include_router(ai_router)


# -------------------------------------------------------------------
# Static dashboard
# -------------------------------------------------------------------
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")


# -------------------------------------------------------------------
# Auth config
# -------------------------------------------------------------------
SECRET_KEY = "change-this-secret-key-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


# -------------------------------------------------------------------
# In-memory user store
# NOTE:
# This is suitable for the first SaaS step and testing.
# Later this should be replaced with a real database.
# -------------------------------------------------------------------
fake_users_db: Dict[str, Dict[str, str]] = {}


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    disabled: bool = False


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class UserInDB(UserResponse):
    hashed_password: str


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_user(email: str) -> Optional[UserInDB]:
    user = fake_users_db.get(email.lower())
    if not user:
        return None
    return UserInDB(**user)


def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    user = get_user(email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user(email)
    if user is None:
        raise credentials_exception
    return user


# -------------------------------------------------------------------
# Root / Health
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------------------------------------------------
# Auth endpoints
# -------------------------------------------------------------------
@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest):
    email = payload.email.lower().strip()

    if len(payload.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )

    if email in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )

    user_record = {
        "email": email,
        "full_name": payload.full_name,
        "disabled": False,
        "hashed_password": get_password_hash(payload.password),
    }

    fake_users_db[email] = user_record

    return UserResponse(
        email=email,
        full_name=payload.full_name,
        disabled=False
    )


@app.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username.lower().strip(), form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer"
    )


@app.get("/me", response_model=UserResponse)
def read_me(current_user: UserInDB = Depends(get_current_user)):
    return UserResponse(
        email=current_user.email,
        full_name=current_user.full_name,
        disabled=current_user.disabled
    )