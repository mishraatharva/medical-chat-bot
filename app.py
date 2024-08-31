# IMPORT STATEMENTS
from flask import Flask, render_template, jsonify, request, session, redirect
from flask_session import Session
from src.helper import download_hugging_face_embedding, setup_pinecone, get_structured_data
from src.prompt import contextualize_q_prompt,qa_prompt
from dotenv import load_dotenv
from src.prompt import *
import uuid
import os


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain_groq import ChatGroq

from langchain.chains import create_history_aware_retriever, create_retrieval_chain


app = Flask(__name__)
# app.config["SESSION_PERMANENT"] = False
# app.config["SESSION_TYPE"] = "filesystem"
# Session(app)

# if "store" not in Session:
#     Session['store'] = {}



# SETUP API_KEYS
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
groq_api_key=os.getenv("GROQ_API_KEY")




# SETUP MODEL AND PINECONE INDEX
index_name = "medical-chat-bot"
embedding = download_hugging_face_embedding()
db = setup_pinecone(PINECONE_API_KEY,index_name,embedding)
retriver = db.as_retriever(search_type="mmr",search_kwargs={"k": 2})
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


# SESSION_IDS
session_id = uuid.uuid1().hex
store = {}

# SETUP RETRIVERS AND PROMPTS FOR CHAT & MESSAGE HISTORY FUNCTIONALITY
history_aware_retriever=create_history_aware_retriever(model,retriver,contextualize_q_prompt)
question_answer_chain=create_stuff_documents_chain(model,qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

def get_session_history(session:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]
    

conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

#**********************************************************************************FLASK CODE******************************************************************************#

# ROUTE FOR INDEX PAGE
@app.route("/")
def index():
    return render_template("chat.html")


# ROUTE TO TAKE QUERY FROM USER AND RETURN ANSWER TO USER
@app.route("/get", methods=["GET", "POST"])
def chat():
    final_result = ""
    msg = request.form["msg"]
    input = msg
    print(input)
    try:
        print("try")
        result = conversational_rag_chain.invoke({"input": input}
                                                 ,config={"configurable": {"session_id":session_id}})
        final_result = get_structured_data(result)
        print(final_result)
    except Exception as e:
        print("except")
        final_result = {"result":"some error occured!"}
    return str(final_result)


# APPLICATION START 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True) 