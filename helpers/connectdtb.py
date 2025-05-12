from pymongo import MongoClient
from pymongo.errors import PyMongoError

def get_mongo_client(uri="mongodb+srv://TungConnectDTB:TungConnectDTB@cluster0.berquuj.mongodb.net/", db_name="MusicModel"):
  try:
    client = MongoClient(uri)
    db = client[db_name]
    print("Connected to MongoDB successfully!")
    return db
  except PyMongoError as e:
    print(f"Error connecting to MongoDB: {e}")
    return None