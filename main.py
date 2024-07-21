from fastapi import FastAPI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Langchain + Pinecone Imports
from langchain.chat_models.openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings

app = FastAPI()
load_dotenv()

# CORS Setup
origins = [
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
llm = ChatOpenAI(temperature=0.7, openai_api_key=os.environ['OPENAI_API_KEY'])
embed = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1536)
vectstr = PineconeVectorStore(
    pinecone_api_key=os.getenv('PINECONE_API_KEY'),
    embedding=embed,
    index_name='my-site-index'
)
# API ENDPOINTS ---------------------------------------------

@app.get("/")
def read_init():
    if vectstr == None:
        return {'status': 404}
    return {'status': 200}

@app.post("/search")
def read_search(item: Item):
    if item.query == None:
        return {'status': 404, 'response': 'No query provided'}
    
    context = ",".join([d.page_content for d in vectstr.similarity_search(item.query, k = 3)])
    prompt = PromptTemplate.from_template(
        "Pretend you're Akhter (Nawid) Tahmid. Speak professionally. No complicated words. Answer in few short sentences: {question}?. Here is Nawid's relevant past: " + context
    )
    chain = prompt | llm | StrOutputParser()

    chain.invoke({
        "question": item.query
})
