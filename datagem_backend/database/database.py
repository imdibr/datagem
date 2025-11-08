import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# =====================
# DATABASE URL CONFIGURATION
# =====================
# Priority:
# 1. DATABASE_URL environment variable (for production/PostgreSQL like Render)
# 2. Default to SQLite for local development
# =====================

# Get the base directory (datagem_backend folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# Check for DATABASE_URL environment variable first (Render sets this automatically)
DATABASE_URL = os.getenv("DATABASE_URL")

# If no DATABASE_URL is set, use SQLite for local development
if not DATABASE_URL:
    # SQLite database file path (relative to datagem_backend directory)
    db_path = BASE_DIR / "datagem.db"
    DATABASE_URL = f"sqlite:///{db_path}"
    print(f"Using SQLite database at: {db_path}")
else:
    print(f"Using database from DATABASE_URL environment variable")

# 1. Create the main "engine" (the power plug)
# For SQLite, we need to add check_same_thread=False for FastAPI compatibility
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL query logging
    )
else:
    # For PostgreSQL and other databases
    engine = create_engine(DATABASE_URL, echo=False)

# 2. Create a "factory" that makes new database conversations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. Create a base "blueprint" for all our database tables (models)
Base = declarative_base()

# 4. This is the helper that gives a database conversation to each API request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
