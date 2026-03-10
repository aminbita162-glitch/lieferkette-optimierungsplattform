from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models.user import User
from app.models.warehouse import Warehouse
from app.models.order import Order
from app.routers.ai_router import router as ai_router
from app.routers.optimize import router as optimize_router
from app.routers.simulation_router import router as simulation_router
from app.routers.warehouse_router import router as warehouse_router
from app.routers.order_router import router as order_router


# ---------------------------------
# Database init
# ---------------------------------
# Important:
# We recreate only the orders table once so the updated schema
# (including nullable optimized_route) is applied in Render DB.
Order.__table__.drop(bind=engine, checkfirst=True)
Order.__table__.create(bind=engine, checkfirst=True)

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Lieferkette Optimierungsplattform API",
    version="0.8.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

SECRET_KEY = "change-this-secret-key-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


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


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_user(email: str, db: Session) -> Optional[UserInDB]:
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user:
        return None

    return UserInDB(
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        hashed_password=user.hashed_password,
    )


def authenticate_user(email: str, password: str, db: Session) -> Optional[UserInDB]:
    user = get_user(email, db)
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
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> UserInDB:
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

    user = get_user(email, db)
    if user is None:
        raise credentials_exception
    return user


app.include_router(optimize_router)
app.include_router(simulation_router)
app.include_router(ai_router)
app.include_router(warehouse_router)
app.include_router(order_router)


@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()

    if len(payload.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )

    new_user = User(
        email=email,
        full_name=payload.full_name,
        hashed_password=get_password_hash(payload.password),
        disabled=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse(
        email=new_user.email,
        full_name=new_user.full_name,
        disabled=new_user.disabled
    )


@app.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(form_data.username.lower().strip(), form_data.password, db)

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