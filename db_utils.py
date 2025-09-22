from pymongo.mongo_client import MongoClient
from pymongo.errors import PyMongoError
import os

# Get the MongoDB connection string from the environment file.
# mongo_uri = os.getenv("MONGODB_CONNECTION_STRING")
mongo_uri = "mongodb+srv://anujsharma:anujmongo@cluster0.ll7dbi2.mongodb.net"

# Define the name of the database and collection to use.
# IMPORTANT: Replace 'your_existing_database_name' with your actual database name.
DATABASE_NAME = "pokemon_rag"
COLLECTION_NAME = "pokemon_embeddings"

def get_mongo_client():
    """
    Establishes and returns a connection to the MongoDB Atlas cluster.
    """
    if not mongo_uri:
        print("MongoDB connection string not found in .env file.")
        print(f"Current value of mongo_uri: {mongo_uri}")
        raise ValueError("MongoDB connection string not found in .env file.")

    try:
        print(f"Attempting to connect to MongoDB with URI: {mongo_uri}")
        client = MongoClient(mongo_uri)
        # Ping the database to confirm a successful connection.
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except PyMongoError as e:
        print(f"Failed to connect to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_pokemon_collection():
    """
    Returns a reference to the Pokémon collection within the specified database.
    """
    client = get_mongo_client()
    if client:
        db = client[DATABASE_NAME]
        return db[COLLECTION_NAME]
    return None

if __name__ == "__main__": 
    # Test the connection and collection retrieval.
    collection = get_pokemon_collection()
    if collection:
        print(f"Successfully accessed collection: {COLLECTION_NAME} in database: {DATABASE_NAME}")
    else:
        print("Failed to access the Pokémon collection.")