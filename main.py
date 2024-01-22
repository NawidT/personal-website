from fastapi import FastAPI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Langchain + Pinecone Imports
from langchain import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

app = FastAPI()
load_dotenv()

# CORS Setup
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5173",
    "https://resume-nawidt.vercel.app",
    "https://resume-sigma-eosin.vercel.app",
    "https://resume-sigma-eosin.vercel.app/",
    "https://nawidtahmid.vercel.app/",
    "https://nawidtahmid.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request Body Paramater
class Item(BaseModel):
    query: str

# starter code
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='text-embedding-ada-002'    
)

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

# getting the index
index = os.getenv('PINECONE_INDEX')
pc_vecstr = Pinecone.from_existing_index(index_name=index, embedding=embeddings)

# creating the prompt for second chain
prompt = PromptTemplate(
    input_variables=["question"],
    template="Pretend you're Akhter (Nawid) Tahmid. Speak professionally. No complicated words. Answer in few short sentences: {question}?"
)

llm = OpenAI(temperature=0.7, max_tokens=500)

# creating first chain for retrieval
retr_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=pc_vecstr.as_retriever()
)

# creating second chain for answering
ans_chain = LLMChain(
    llm=llm,
    prompt=prompt,

)


# API ENDPOINTS ----------------------------------------------

@app.get("/")
def read_init():
    if pc_vecstr == None:
        return {'status': 404}
    return {'status': 200}

@app.post("/search")
def read_search(item: Item):
    if item.query == None:
        return {'status': 404, 'response': 'No query provided'}
    retr_feed = prompt.format(question=item.query)
    ans = retr_chain.run(retr_feed)
    return {'status': 200, 'response': ans}
