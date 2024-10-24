import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("openai_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model_name="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

def rag(query):
    system_prompt = '''Your task is to act as our website helper bot. We will ask you any query related to our website 'Solvecube' and you will answer smartly.\n "{context}"'''
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    retriever = db.as_retriever()

    # creating rag_chain with history_aware_retriever.. retriever get context from embedded documents based on rephrased query and then add this context to system prompt and then send it to llm
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    result = response["answer"]

    return result

st.header("SolveCube Chatbot")

# text input field
user_query = st.text_input(label="", help="Ask here to learn about SolveCube", placeholder="What do you want to know about SolveCube?")

rag_response = rag(user_query)
st.header("My Response")
st.markdown(rag_response)
