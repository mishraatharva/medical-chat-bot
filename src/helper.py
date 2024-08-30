# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings

# NEW IMPORTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents


# Create Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


#Download HuggingFaceEmbedding Model
def download_hugging_face_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


# Authenticate with pinecone
def setup_pinecone(api_key,index_name,embedding):
    pc = Pinecone(api_key="f5332a0e-31e2-49be-8512-cd45f97e31e0")
    index = pc.Index(index_name)
    # db=Pinecone.from_existing_index(index_name, embedding)

    db = PineconeVectorStore(index=index, embedding=embedding,index_name=index_name)
    return db


# get chain
def get_retriver_chain(model,prompt,db):
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 2}), question_answer_chain)
    return chain


# format data as per our reqirement
def get_structured_data(result):
    answer = result["answer"]
    source1 = result['context'][0].metadata['page']
    source2 = result['context'][1].metadata['page']

    return f"answer: {answer}\n\nsource: {source1} and {source2}"