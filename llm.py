from typing import Dict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Validate environment
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set.")
    st.stop()

# Initialize Groq LLM
try:
    llm = ChatGroq(model_name="gemma2-9b-it", api_key=GROQ_API_KEY, temperature=0.7)
except Exception as e:
    st.error(f"Failed to initialize Groq LLM: {e}")
    st.stop()
