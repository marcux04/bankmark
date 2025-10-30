from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
print("Bases de datos:", client.list_database_names())

db = client["bank_marketing_db"]
print("Colecciones:", db.list_collection_names())
