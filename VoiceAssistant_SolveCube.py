import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import streamlit as st
import noisereduce as nr
from pydub import AudioSegment
import speech_recognition as sr
import numpy as np
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from gtts import gTTS
from io import BytesIO
import pygame 

# initiate pygame
pygame.mixer.init()

def text_to_speech(text):
    # Convert text to speech using gTTS and return the audio as an AudioSegment object
    tts = gTTS(text, lang='en', slow=False)  # You can change the language by altering the 'lang' argument
    mp3_fp = BytesIO()  # Use a BytesIO object to store the audio in memory (no saving to disk)
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    pygame.mixer.music.load(mp3_fp)
    return mp3_fp

def play_audio_segment():
    # Function to play audio, will be run in a separate process
    pygame.mixer.music.play()

def stop_speech():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


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
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        print(transcription)
        return transcription.text

# Load embeddings and models
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("openai_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model_name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

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

    if len(result.split()) > 1000:
        result = "The response is too long. Please refine your question or try again."
    
    return result

def main():
    st.title("Solvecube Chatbot")

    text_input = st.text_input(label="Ask here to learn about SolveCube", help="Ask here to learn about SolveCube", placeholder="What do you want to know about SolveCube?")
    if text_input:
        print(text_input)
        print(st.session_state)
        response = rag(text_input)
        if 'audio_playing' not in st.session_state:
            st.session_state.audio_playing = True 
        # Stop any ongoing audio before playing new audio
        stop_speech()
        # Play new audio in a separate process
        text_to_speech(response) 
        st.markdown(response)
        play_audio_segment()

    if 'audio_playing' not in st.session_state:
        st.session_state.audio_playing = False  # Initialize the state

    stop_button = st.button("Stop Audio", key="stop_audio")
    
    if st.session_state.stop_audio:
        print("pressed")
        stop_speech()
        st.session_state.audio_playing = False  # Manage the state

    button_pressed = st.button("Record Audio", key="record_audio")
    
    if button_pressed:
        transcription = record_audio()
        response = rag(transcription)
        # Stop any ongoing audio before playing new audio
        stop_speech()
        # Play new audio in a separate process
        text_to_speech(response) 
        st.markdown(response)
        play_audio_segment()
        st.session_state.audio_playing = True  # Manage the state

    # Manage button styling
    if st.session_state.audio_playing:
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
