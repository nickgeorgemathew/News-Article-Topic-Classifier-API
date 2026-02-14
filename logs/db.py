import json
import os
import sqlite3
from datetime import datetime
from sqlalchemy.orm import declarative_base,sessionmaker
from sqlalchemy import create_engine
from pathlib import Path

DB_path=os.environ.get("DB_PATH","./logs/logs.db")
DATABASE_URL="sqlite:///./logs.db"

engine=create_engine(
    f"sqlite:///{DB_path}",connect_args={"check_same_thread":False}
)
sessionlocal=sessionmaker(bind=engine)
Base=declarative_base()