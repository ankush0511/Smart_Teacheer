import streamlit as st
from yt_transcript_RAG.utills.youtube_utils import extract_video_id, get_transcript_and_summary
from yt_transcript_RAG.utills.pinecone_utils import create_pinecone_index, index_documents, retrieve_documents
from yt_transcript_RAG.utills.llm_utils import get_embedding, generate_answer
from dataclasses import dataclass
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FetchedTranscriptSnippet:
    text: str
    start: float
    duration: float

def main():
    st.set_page_config(page_title="YouTube Transcript & RAG System", layout="wide")
    st.title("YouTube Transcript and RAG Query System")

    # Initialize session state
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'index' not in st.session_state:
        st.session_state.index = create_pinecone_index()

    # Input for YouTube URL
    youtube_url = st.text_input("Paste YouTube URL here:", placeholder="https://youtu.be/...")
    if st.button("Process Video") and youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.session_state.video_id = video_id
            with st.spinner("Fetching transcript and generating summary..."):
                try:
                    transcript, summary = get_transcript_and_summary(video_id)
                    st.session_state.transcript = transcript
                    st.session_state.summary = summary
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                    documents = text_splitter.create_documents([summary])
                    index_documents(documents, st.session_state.index)
                    st.success("Video processed successfully!")
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    logger.error(f"Error processing video: {e}")

    # Display transcript and summary
    if st.session_state.transcript and st.session_state.summary:
        st.subheader("Summary")
        st.text_area("Summary", st.session_state.summary, height=400)

    # RAG Query System
    st.subheader("Ask Questions About the Video")
    query = st.text_input("Enter your question:", placeholder="Why is RAG used?")
    if st.button("Get Answer") and query and st.session_state.video_id:
        with st.spinner("Generating answer..."):
            try:
                retrieved_docs = retrieve_documents(query, st.session_state.index)
                answer = generate_answer(query, retrieved_docs)
                st.write("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                logger.error(f"Error generating answer: {e}")

    st.sidebar.info("Paste a new YouTube URL to process a different video. The system will update with the new content.")

if __name__ == "__main__":
    main()