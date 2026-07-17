from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models.user import User
from app.models.activity_log import ActivityLog
from app.schemas.user_schema import UserSignupSchema, UserResponseSchema, TokenResponseSchema
from app.auth.utils import get_password_hash, verify_password, create_access_token
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

@router.post("/signup", response_model=UserResponseSchema, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserSignupSchema):
    """Registers a new user in the system (Admin, Manager, Developer, or Client)."""
    # Check if user already exists
    existing_user = await User.find_one(User.email == payload.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists."
        )
    
    # Hash password and save user
    hashed_pw = get_password_hash(payload.password)
    new_user = User(
        name=payload.name,
        email=payload.email,
        hashed_password=hashed_pw,
        role=payload.role
    )
    await new_user.insert()

    # Log user registration activity
    activity = ActivityLog(
        user_id=str(new_user.id),
        user_name=new_user.name,
        action="user_registered",
        detail=f"User registered: {new_user.name} ({new_user.email}) as role {new_user.role}."
    )
    await activity.insert()

    return new_user

@router.post("/login", response_model=TokenResponseSchema)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Validates login credentials and returns a secure JWT token."""
    user = await User.find_one(User.email == form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if user.status != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is inactive"
        )
    
    access_token = create_access_token(data={"sub": user.email, "role": user.role})

    # Log successful login activity
    activity = ActivityLog(
        user_id=str(user.id),
        user_name=user.name,
        action="user_login",
        detail=f"User {user.name} logged in successfully."
    )
    await activity.insert()

    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logs the user logout event in activity logs."""
    activity = ActivityLog(
        user_id=str(current_user.id),
        user_name=current_user.name,
        action="user_logout",
        detail=f"User {current_user.name} logged out."
    )
    await activity.insert()
    return {"detail": "Logged out successfully"}
