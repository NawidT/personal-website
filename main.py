from fastapi import FastAPI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Langchain + Pinecone Imports
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.chains.question_answering import load_qa_chain

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
    "https://resume-sigma-eosin.vercel.app/"
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
    query: str | None = None

# starter code
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='text-embedding-ada-002'    
)
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
index = os.getenv('PINECONE_INDEX')
docsem = Pinecone.from_existing_index(index_name=index, embedding=embeddings)
prompt_template = "Pretend you are Akhter (Nawid) Tahmid. Speak in a professional manner, but don't use complicated words. Don't use information outside of whats given. Answer the following question in a couple short sentences: {question}?"
llm = ChatOpenAI(
    temperature=0, 
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-3.5-turbo'
)

qa_chain = load_qa_chain(
    llm=llm, 
    chain_type="stuff",
)

@app.get("/")
def read_init():
    if docsem == None:
        return {'status': 404}
    return {'status': 200}

@app.post("/search")
def read_search(item: Item):
    if item.query == None:
        return {'status': 404, 'response': 'No query provided'}
    docs = docsem.similarity_search(item.query) 
    ans = qa_chain.run(input_documents=docs, question=prompt_template.format(question=item.query))
    return {'status': 200, 'response': ans}
