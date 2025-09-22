import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

print("Let's Begin with the rag application")
print("Loading models and API keys")

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except TypeError:
    print("Error in api key")
    exit()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = genai.GenerativeModel('gemini-pro')

print("Conneting to the vector databases...")
try:
    client = chromadb.PersistentClient(path="pokemon_db")
    collection = client.get_collection(name="pokemon_embeddings")
    print(f"Successfully connected to the vector databases! Got {collection.count()} items.")
except Exception as e:
    print(f"Error connecting to the vector databases: {e}")
    print(f"Have you run the embed_data.py")
    exit()


    # LLets get the required documents via queries

def retrieve_relevant_documents(query):
        """
    This function takes a user's question, embeds it, and retrieves
    the most relevant documents from the ChromaDB collection.
        """

        print(f"\n Lets Go step by step \n 1) Embedding the Query: '{query}'")
        query_embedding = embedding_model.encode(query)

        print("\n 2) Searching the DB for relevant docs...")
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )

        retrieved_docs = results['documents'][0]
        print("\n3) Found Relevant Docs")
        return retrieved_docs
    
if(__name__ == "__main__"):
        print("Lets start the Pokemon Q/A");
        user_query = input("Enter your question: ")
        retrieved_context = retrieve_relevant_documents(user_query)

        print("\nRaw Retrieved Docs")
        for i, doc in enumerate(retrieved_context):
            print(f"Document{i+1}: \n{doc}\n")