from dotenv import load_dotenv
load_dotenv()
import os
import json
import uuid
from openai import OpenAI
from pinecone import Pinecone

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

print(pinecone_api_key)

default_index = "home-task-chatbot"
default_namespace = "home-task-namespace"
data_file_path = "./data.json"

pinecone = Pinecone(api_key=pinecone_api_key)
index = pinecone.Index(default_index)

openai = OpenAI(api_key=openai_api_key)

def load_data_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def store_data_in_pinecone(file_path):
    data = load_data_from_json(file_path)

    for item in data:
        question = item["question"]
        answer = item["answer"]

        embedding = get_embedding(question)
        
        unique_id = str(uuid.uuid4())

        index.upsert(
            vectors=[
                {
                    "id": unique_id,
                    "values": embedding,
                    "metadata": {
                        "question": question,
                        "answer": answer
                    }
                }
            ],
            namespace=default_namespace
        )

store_data_in_pinecone(data_file_path)

# response = index.query(
#     namespace=default_namespace,
#     vector=[0.1, 0.3],
#     top_k=2,
#     include_values=True,
#     include_metadata=True,
#     filter={"genre": {"$eq": "action"}}
# )

# print(response)