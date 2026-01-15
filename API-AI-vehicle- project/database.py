import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Database URL
SQLALCHEMY_DATABASE_URL = os.getenv("DB_URL")

if not SQLALCHEMY_DATABASE_URL:
    # Fallback to SQLite if DB_URL is not set (for local testing without .env)
    print("⚠️ WARNING: DB_URL not found. Database connection might fail.")
    SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db" 

# Auto-fix for Aiven/MySQL URLs (SQLAlchemy requires 'mysql+pymysql')
if SQLALCHEMY_DATABASE_URL.startswith("mysql://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)

# Create the database engine
try:
    # pool_recycle=3600 is required for MySQL to prevent connection timeouts
    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_recycle=3600)
except Exception as e:
    print(f"❌ Engine Creation Error: {e}")
    raise e

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for models
Base = declarative_base()

# Connection Test
try:
    with engine.connect() as connection:
        print("✅ Connection Successful! Database is accessible.")
except Exception as e:
    print(f"❌ Connection Failed: {e}")