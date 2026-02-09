from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from models import user, scenario, simulation, decision, behavior_profile
    Base.metadata.create_all(bind=engine)

    # Add gemini_cache column if missing (for existing databases)
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("simulations")]
    if "gemini_cache" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE simulations ADD COLUMN gemini_cache JSONB"))
            conn.commit()
