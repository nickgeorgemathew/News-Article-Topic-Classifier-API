import json
import sqlite3
from datetime import datetime
from sqlalchemy.orm import declarative_base,sessionmaker
from sqlalchemy import create_engine
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
DB_path=BASE_DIR / "logs.db"
DATABASE_URL="sqlite:///./logs.db"

engine=create_engine(
    f"sqlite:///{DB_path}",connect_args={"check_same_thread":False}
)
sessionlocal=sessionmaker(bind=engine)
Base=declarative_base()