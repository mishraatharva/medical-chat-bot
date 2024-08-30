from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding, setup_pinecone, get_structured_data, get_retriver_chain
# import pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


from src.prompt import prompt_template

# from langchain_pinecone import Pinecone
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
groq_api_key=os.getenv("GROQ_API_KEY")


# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# chain_type_kwargs={"prompt": PROMPT}

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)




# ## PINECONE CODE START
# pc = Pinecone(api_key="f5332a0e-31e2-49be-8512-cd45f97e31e0")
# index = pc.Index(index_name)
# # db=Pinecone.from_existing_index(index_name, embedding)
# db = PineconeVectorStore(index=index, embedding=embedding,index_name=index_name)

# ## PINECONE CODE END

index_name = "medical-chat-bot"
embedding = download_hugging_face_embedding()
db = setup_pinecone(PINECONE_API_KEY,index_name,embedding)
print(db)

model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

# qa=RetrievalQA.from_chain_type(
#     llm=model, 
#     chain_type="stuff", 
#     retriever=db.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True, 
#     chain_type_kwargs=chain_type_kwargs)


# question_answer_chain = create_stuff_documents_chain(model, prompt)
# chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 2}), question_answer_chain)

chain = get_retriver_chain(model,prompt,db)

final_result = ""
#**********************************************************************************FLASK CODE******************************************************************************#
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    try:
        result = chain.invoke({"input": input})
        print("---------------------")
        print(result)
        final_result = get_structured_data(result)
        print(final_result)
        print("----------------------")
    except Exception as e:
        result = {"result":"some error occured!"}
        print("Response : ", result)

    return str(final_result)


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     try:
#         result = qa.invoke(input)
#     except Exception as e:
#         result = {"result":"some error occured!"}
#         print("Response : ", result)

#     return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
















# llm = None
# path_to_model = r'C:\Users\Naruto\Desktop\generative_ai\generative_ai_material\project\Medical_Chat_Bot\model\llama-2-7b-chat.ggmlv3.q4_0.bin'
# llm = CTransformers(
#     model=path_to_model,
#     model_type='llama',
#     config={'max_new_tokens':552,
#             'temperature':0.8}
#     )    