from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os, json, faiss
import numpy as np
from dotenv import load_dotenv

origins = [
    "http://localhost:3000",
    "http://localhost:3000/"
]

load_dotenv(override=True)
client = OpenAI(api_key=os.environ['OPENAIKEY'])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def askUSC(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful, polite, and succinct assistant to students at the University of Southern California."},
        {"role": "user", "content": prompt}
        ])
    print(completion)
    response = completion.choices[0].message.content
    return response

def generate_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def retrieve_chunks(indices):
    with open('chunks.json', 'r') as infile:
        all_chunks = json.load(infile)
    return [all_chunks[i] for i in indices[0]]

# def query_vector_store(query, index_path='student_policy_embeddings.index', k=5):

def query_vector_store(query, index_path='usc_embeddings.index', k=5):
    index = faiss.read_index(index_path)
    query_embedding = generate_embeddings(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding_np, k)
    print(distances)
    return indices

def create_context(query):
    indices = query_vector_store(query)
    context_chunks = retrieve_chunks(indices)
    context = " ".join(context_chunks)
    return context

def generate_answer(query):
    context = create_context(query)
    # print(context)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a special student policy chatbot for students at the University of Southern California. Don't answer anything that is not related to USC."},
            {"role": "user", "content": f"Context:\n{context}\nQuestion:\n{query}"}
        ],
        max_tokens=150,
        temperature=0.2
    )
    
    return response.choices[0].message.content

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ask_usc")
async def ask_usc(request: Request):
    request_json = await request.body()
    prompt = request_json.decode('utf-8')
    response = askUSC(prompt)
    return response

@app.post("/search")
async def search(request: Request):
    request_json = await request.body()
    prompt = request_json.decode('utf-8')
    print(prompt)
    response = generate_answer(prompt)
    return response


