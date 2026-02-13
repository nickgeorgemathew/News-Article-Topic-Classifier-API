from sqlalchemy import Column,Integer,DateTime,String,JSON,Float
from datetime import datetime
from db import engine
from db import Base

class Predictionlog(Base):
    __tablename__="predictionlogs"

    id=Column(Integer,primary_key=True,index=True,autoincrement=True)
    timestamp=Column(DateTime)
    Text=Column(String,index=True)
    top_prediction=Column(String)
    top_score=Column(Float)
    all_scores=Column(JSON)
    model_version=Column(String)
    error=Column(String,nullable=True)


Base.metadata.create_all(bind=engine)