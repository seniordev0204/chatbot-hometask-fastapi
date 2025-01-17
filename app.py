from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()
import os

from openai import OpenAI
from pinecone import Pinecone

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

default_index = "home-task-chatbot"
default_namespace = "home-task-namespace"

pinecone = Pinecone(api_key=pinecone_api_key)
index = pinecone.Index(default_index)

openai = OpenAI(api_key=openai_api_key)

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def create_prompt(query_result, user_question):
    """
    Create a prompt for GPT-4 using the query results and the user's question.
    
    Args:
        query_result (dict): The results from the Pinecone index query.
        user_question (str): The current question from the user.
    
    Returns:
        str: A formatted prompt for GPT-4.
    """
    # Start the prompt
    prompt = "You are a helpful assistant. Below is relevant information from a database:\n\n"
    
    # Add the top matches to the prompt
    for idx, match in enumerate(query_result.get('matches', []), start=1):
        metadata = match.get('metadata', {})
        retrieved_question = metadata.get('question', "Unknown question")
        retrieved_answer = metadata.get('answer', "Unknown answer")
        prompt += f"{idx}. Q: {retrieved_question}\n   A: {retrieved_answer}\n\n"
    
    # Include the user's question
    prompt += "Using the above information, answer the following question:\n"
    prompt += f"Q: {user_question}\n\n"
    prompt += "Provide a concise and accurate response."
    
    return prompt

def query_gpt4(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", "content": "You are a helpful assistant."
            },
            {
                "role": "user", "content": prompt
            }
        ]
    )
    return response.choices[0].message.content

class QuestionRequest(BaseModel):
    question: str

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],  # You can specify allowed methods instead of "*"
    allow_headers=["*"],  # You can specify allowed headers instead of "*"
)

@app.post('/ask')
async def handle_question(request: QuestionRequest):
    question = request.question

    result = index.query(
        namespace=default_namespace,
        vector=get_embedding(question),
        top_k=4,
        include_values=True,
        include_metadata=True,
    )

    prompt = create_prompt(result, question)
    answer = query_gpt4(prompt)

    response = {
        "question": question,
        "answer": answer
    }

    return response
