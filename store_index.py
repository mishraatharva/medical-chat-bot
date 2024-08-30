from pinecone import Pinecone
from src.helper import load_pdf, text_split, download_hugging_face_embedding

from dotenv import load_dotenv
import os

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# index = os.environ.get('INDEX_NAME')


# pc = Pinecone(api_key=PINECONE_API_KEY)

def get_util_variables():
    load_dotenv()

    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    print(PINECONE_API_KEY)
    index_name = os.environ.get('INDEX_NAME')
    print(index_name)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    print(index)
    return index

# get_util_variables()

def push_data_to_pinecone_index_util():
    extracted_data = load_pdf("data/")
    text_chunks = text_split(extracted_data)
    return text_chunks


def push_data_to_pinecone_index():
    
    text_chunks = push_data_to_pinecone_index_util()
    embedding = download_hugging_face_embedding()
    index = get_util_variables()

    for i, t in zip(range(len(text_chunks)), text_chunks):
        query_result = embedding.embed_query(t.page_content)
        index.upsert(
        vectors=[
            {
                "id": str(i),  # Convert i to a string
                "values": query_result, 
                "metadata": {"text":str(text_chunks[i].page_content)} # meta data as dic
            }
        ],
        namespace="real" 
        )

        index.describe_index_stats() 
        
