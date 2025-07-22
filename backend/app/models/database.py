from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# SQLAlchemy Base
Base = declarative_base()

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    solar_panels = relationship("SolarPanel", back_populates="user", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="user", cascade="all, delete-orphan")

class SolarPanel(Base):
    __tablename__ = "solar_panels"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    address = Column(Text)
    capacity = Column(Float)  # in kW
    installation_date = Column(DateTime)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="solar_panels")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    image_path = Column(Text, nullable=False)
    prediction = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(50))
    processing_time = Column(Float)  # in seconds
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="analysis_results")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("prediction IN ('clean', 'dirty')", name="valid_prediction"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="valid_confidence"),
    )

# Pydantic Models for API
class UserBase(BaseModel):
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    role: str = Field(default="user", description="User role")

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="User password")

class UserResponse(UserBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class AnalysisResultBase(BaseModel):
    image_path: str = Field(..., description="Path to the uploaded image")
    prediction: str = Field(..., description="Prediction result (clean/dirty)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    model_version: Optional[str] = Field(None, description="Model version used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class AnalysisResultCreate(AnalysisResultBase):
    user_id: Optional[UUID] = Field(None, description="User ID who made the analysis")

class AnalysisResultResponse(AnalysisResultBase):
    id: UUID
    user_id: Optional[UUID]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class SolarPanelBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Solar panel name")
    latitude: Optional[float] = Field(None, description="Latitude")
    longitude: Optional[float] = Field(None, description="Longitude")
    address: Optional[str] = Field(None, description="Address")
    capacity: Optional[float] = Field(None, description="Panel capacity in kW")
    installation_date: Optional[datetime] = Field(None, description="Installation date")

class SolarPanelCreate(SolarPanelBase):
    user_id: UUID = Field(..., description="User ID who owns the panel")

class SolarPanelResponse(SolarPanelBase):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# Database Manager
class DatabaseManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_user(self, user_data: dict) -> UserResponse:
        """Create a new user"""
        user = User(**user_data)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return UserResponse.model_validate(user)

    async def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get user by email"""
        from sqlalchemy import text
        result = await self.session.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": email}
        )
        user = result.fetchone()
        if user:
            return UserResponse.model_validate(user)
        return None

    async def get_user_by_id(self, user_id: UUID) -> Optional[UserResponse]:
        """Get user by ID"""
        from sqlalchemy import text
        result = await self.session.execute(
            text("SELECT * FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        )
        user = result.fetchone()
        if user:
            return UserResponse.model_validate(user)
        return None

    async def create_analysis_result(self, result_data: dict) -> AnalysisResultResponse:
        """Create a new analysis result"""
        analysis = AnalysisResult(**result_data)
        self.session.add(analysis)
        await self.session.commit()
        await self.session.refresh(analysis)
        return AnalysisResultResponse.model_validate(analysis)

    async def get_user_analysis_results(self, user_id: UUID, limit: int = 10) -> List[AnalysisResultResponse]:
        """Get analysis results for a user"""
        from sqlalchemy import text
        result = await self.session.execute(
            text("""
            SELECT * FROM analysis_results 
            WHERE user_id = :user_id 
            ORDER BY created_at DESC 
            LIMIT :limit
            """),
            {"user_id": user_id, "limit": limit}
        )
        analyses = result.fetchall()
        return [AnalysisResultResponse.model_validate(analysis) for analysis in analyses]

    async def create_solar_panel(self, panel_data: dict) -> SolarPanelResponse:
        """Create a new solar panel"""
        panel = SolarPanel(**panel_data)
        self.session.add(panel)
        await self.session.commit()
        await self.session.refresh(panel)
        return SolarPanelResponse.model_validate(panel)

    async def get_user_solar_panels(self, user_id: UUID) -> List[SolarPanelResponse]:
        """Get solar panels for a user"""
        from sqlalchemy import text
        result = await self.session.execute(
            text("SELECT * FROM solar_panels WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        panels = result.fetchall()
        return [SolarPanelResponse.model_validate(panel) for panel in panels]

    async def get_analysis_stats(self, user_id: Optional[UUID] = None) -> dict:
        """Get analysis statistics"""
        if user_id:
            result = await self.session.execute(
                text("""
                SELECT 
                    COUNT(*) as total_analyses,
                    COUNT(CASE WHEN prediction = 'clean' THEN 1 END) as clean_count,
                    COUNT(CASE WHEN prediction = 'dirty' THEN 1 END) as dirty_count,
                    AVG(CASE WHEN prediction = 'clean' THEN confidence END) as avg_confidence_clean,
                    AVG(CASE WHEN prediction = 'dirty' THEN confidence END) as avg_confidence_dirty
                FROM analysis_results 
                WHERE user_id = :user_id
                """),
                {"user_id": user_id}
            )
        else:
            result = await self.session.execute(
                text("""
                SELECT 
                    COUNT(*) as total_analyses,
                    COUNT(CASE WHEN prediction = 'clean' THEN 1 END) as clean_count,
                    COUNT(CASE WHEN prediction = 'dirty' THEN 1 END) as dirty_count,
                    AVG(CASE WHEN prediction = 'clean' THEN confidence END) as avg_confidence_clean,
                    AVG(CASE WHEN prediction = 'dirty' THEN confidence END) as avg_confidence_dirty
                FROM analysis_results
                """)
            )
        
        stats = result.fetchone()
        return {
            "total_analyses": stats.total_analyses or 0,
            "clean_count": stats.clean_count or 0,
            "dirty_count": stats.dirty_count or 0,
            "avg_confidence_clean": float(stats.avg_confidence_clean or 0.0),
            "avg_confidence_dirty": float(stats.avg_confidence_dirty or 0.0)
        } 