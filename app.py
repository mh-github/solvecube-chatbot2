import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("openai_index", embeddings, allow_dangerous_deserialization=True)
llm = OpenAI(model_name="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

def rag(query):
    prompt_template = """

    Human: Use the following pieces of context to provide a detailed respone to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    ##!!! PLAY AROUND WITH SEARCH TYPE AND HOW MANY CHUNKS (SEARCH KWARGS) TO SEND AS CONTEXT TO MODEL
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_type="similarity", search_kwargs={"k": 100}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": query})["result"]

st.header("SolveCube Chatbot")

# text input field
user_query = st.text_input(label="", help="Ask here to learn about SolveCube", placeholder="What do you want to know about SolveCube?")

rag_response = rag(user_query)
st.header("My Response")
st.write(rag_response)
