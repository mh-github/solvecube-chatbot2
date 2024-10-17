import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import noisereduce as nr
from pydub import AudioSegment
import speech_recognition as sr
import numpy as np
import openai
import pyttsx3

engine = pyttsx3.init()

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def stop_speech():
    engine.stop()

def preprocess_audio(audio_data):
    audio_array = np.array(audio_data.get_array_of_samples())
    reduced_noise = nr.reduce_noise(y=audio_array, sr=audio_data.frame_rate)
    processed_audio = AudioSegment(
        reduced_noise.tobytes(), 
        frame_rate=audio_data.frame_rate,
        sample_width=audio_data.sample_width,
        channels=audio_data.channels
    )
    return processed_audio

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        st.write("Recording...")
        audio_data = r.listen(source)
        
        raw_audio = audio_data.get_raw_data()
        
        audio_segment = AudioSegment(
            raw_audio, 
            frame_rate=16000, 
            sample_width=2, 
            channels=1
        )
        processed_audio = preprocess_audio(audio_segment)
        processed_audio.export("processed_audio.wav", format="wav")
        st.write("Transcribing audio with Whisper...")
        with open("processed_audio.wav", "rb") as audio_file:
            transcription = openai.Audio.transcribe(
                model="whisper-1", 
                file=audio_file
            )
        
        return transcription['text']

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("openai_index", embeddings, allow_dangerous_deserialization=True)
llm = OpenAI(model_name="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))

def rag(query):
    prompt_template = """
    Human: Use the following pieces of context to provide a detailed response to the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 10}), 
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa({"query": query})["result"]

    if len(result.split()) > 1000:  
        result = "The response is too long. Please refine your question or try again."
    
    return result

def main():
    st.title("Solvecube Chatbot")

    # Text input field
    text_input = st.text_input(label="", help="Ask here to learn about SolveCube", placeholder="What do you want to know about SolveCube?")
    
    if text_input:
        if "input_field" not in st.session_state or st.session_state.input_field != text_input:
            st.write(rag(text_input))
            text_to_speech(rag(text_input))

    # Stop Audio Button - Always visible
    stopButton = st.button("Stop Audio")

    # Button to record audio
    button_pressed = st.button("Record Audio")
    
    if button_pressed:
        transcription = record_audio()
        st.write(f"{rag(transcription)}")
        text_to_speech(rag(transcription))

    if stopButton:
        stop_speech()

    if 'button_pressed' in st.session_state and st.session_state.button_pressed:
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: #4CAF50; /* Green */
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: #f0f0f0; /* Default color */
                    color: black;
                }
            </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
