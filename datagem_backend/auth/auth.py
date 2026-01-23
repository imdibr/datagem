from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

# We use absolute imports (starting from the root)
from database import crud, database, models as db_models
from auth import models as auth_models
from auth import security # This file now handles hashing



router = APIRouter()

@router.post("/signup", response_model=auth_models.UserInDB)
def create_new_user(user: auth_models.UserCreate, db: Session = Depends(database.get_db)):
    """Create a new user account."""
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # --- THIS IS THE FIX ---
    # We scramble the password here, *before* sending it to crud.py
    hashed_password = security.get_password_hash(user.password)
    db_user = db_models.User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name  # This uses the 'full_name' field
    )
    # --- END FIX ---
    
    return crud.create_user(db=db, user=db_user)


@router.post("/token", response_model=auth_models.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)
):
    """Log in a user and return a JWT access token."""
    user = crud.get_user_by_email(db, email=form_data.username) # username is the email
    
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=auth_models.UserInDB)
async def read_users_me(
    current_user: db_models.User = Depends(security.get_current_active_user),
):
    """Test endpoint to check if a user's token is valid."""
    return current_user
