import os
import logging
import streamlit as st
import wikipediaapi
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Optional
from dotenv import load_dotenv
from YT_transcipt import process_video, query_vector_db
import time
import json
import uuid

# Load environment variables
load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EDUCATION_LEVEL = "college"
COLLECTION_NAME = "academic_data"

# Validate environment
# if not GROQ_API_KEY:
#     st.error("GROQ_API_KEY is not set.")
#     st.stop()

# # Initialize Groq LLM
# try:
#     llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.7, max_tokens=800)
# except Exception as e:
#     st.error(f"Failed to initialize Groq LLM: {e}")
#     st.stop()

# # Initialize ChromaDB and embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_store = Chroma(
#     collection_name=COLLECTION_NAME,
#     embedding_function=embeddings,
#     persist_directory="./chroma_db"
# )

# Initialize Wikipedia API
# wiki = wikipediaapi.Wikipedia("AcademicExplainer/1.0", "en")

def disambiguate_topic(topic: str) -> str:
    """Add context to ambiguous topics for better search results."""
    topic = topic.strip().lower()
    academic_context = {
        "stack": "stack data structure",
        "agile": "agile software development",
        "benzene": "benzene chemistry",
        "attention mechanism": "attention mechanism neural networks",
        "newton third low of motion": "Newton's Third Law of Motion",
        "newton's third law": "Newton's Third Law of Motion"
    }
    return academic_context.get(topic, topic)

# def fetch_wikipedia_explanation(topic: str) -> Optional[str]:
#     """Fetch explanation from Wikipedia."""
#     try:
#         page = wiki.page(topic)
#         if page.exists():
#             summary = page.summary[:1500]  # Increased limit for more context
#             prompt = f"Summarize the following text for a {EDUCATION_LEVEL} student in 150-200 words: {summary}"
#             response = llm.invoke(prompt)
#             return response.content.strip()
#         return None
#     except Exception as e:
#         logger.warning(f"Wikipedia fetch failed for {topic}: {e}")
#         return None

# def fetch_duckduckgo_explanation(topic: str) -> str:
#     """Fetch explanation from DuckDuckGo as fallback."""
#     try:
#         with DDGS() as ddgs:
#             results = list(ddgs.text(f"{topic} explanation", max_results=2))  # Increased results for reliability
#             if results:
#                 content = " ".join([result.get("body", "") for result in results])
#                 prompt = f"Summarize the following text for a {EDUCATION_LEVEL} student in 150-200 words: {content[:1500]}"
#                 response = llm.invoke(prompt)
#                 return response.content.strip()
#             return f"No reliable content found for {topic}."
#     except Exception as e:
#         logger.error(f"DuckDuckGo fetch failed for {topic}: {e}")
#         return f"Error fetching content for {topic}: {e}"

# def fetch_youtube_video(topic: str) -> Dict:
#     """Fetch YouTube video link using DuckDuckGo."""
#     try:
#         with DDGS() as ddgs:
#             results = list(ddgs.videos(f"{topic} tutorial", max_results=1))
#             if results:
#                 video = results[0]
#                 return {
#                     "url": video.get("content", ""),
#                     "title": video.get("title", "Unknown"),
#                     "description": video.get("description", "")
#                 }
#             return {}
#     except Exception as e:
#         logger.error(f"YouTube video fetch failed for {topic}: {e}")
#         return {}

# def process_syllabus(topics: List[str]) -> List[Dict]:
#     """Process syllabus topics for explanations and YouTube links."""
#     results = []
#     for topic in topics:
#         topic = disambiguate_topic(topic)
#         explanation = fetch_wikipedia_explanation(topic) or fetch_duckduckgo_explanation(topic)
#         video_data = fetch_youtube_video(topic)
        
#         doc_id = str(uuid.uuid4())
#         document = Document(
#             page_content=f"Topic: {topic}\nExplanation: {explanation}",
#             metadata={"type": "topic", "topic": topic, "video_url": video_data.get("url", ""), "video_title": video_data.get("title", "")},
#             id=doc_id
#         )
#         vector_store.add_documents([document])
        
#         results.append({
#             "topic": topic,
#             "explanation": explanation,
#             "video_url": video_data.get("url", "No video found"),
#             "video_title": video_data.get("title", "Unknown")
#         })
#         time.sleep(2)
    
#     return results

# def process_youtube_video(video_url: str, title: str = "Unknown") -> Dict:
#     """Process a YouTube video for transcript and summary."""
#     try:
#         result = process_video(video_url, title)
#         if result["stored"]:
#             doc_id = str(uuid.uuid4())
#             document = Document(
#                 page_content=f"Video URL: {video_url}\nTranscript: {result['transcript']}\nSummary: {result['summary']}",
#                 metadata={"type": "video", "topic": title.lower(), "video_url": video_url, "video_title": title},
#                 id=doc_id
#             )
#             vector_store.add_documents([document])
#         return result
#     except Exception as e:
#         logger.error(f"Error processing YouTube video {video_url}: {e}")
#         return {
#             "video_url": video_url,
#             "transcript": f"Error generating transcript: {e}",
#             "summary": f"Error generating summary: {e}",
#             "stored": False
#         }

def fetch_topic_history() -> List[str]:
    """Retrieve unique topics from ChromaDB."""
    try:
        results = vector_store.get()
        topics = set()
        for metadata in results.get("metadatas", []):
            topic = metadata.get("topic")
            if topic:
                topics.add(topic.lower())
        return sorted(list(topics))
    except Exception as e:
        logger.error(f"Failed to fetch topic history: {e}")
        return []

# def generate_mcqs(topic: str, num_questions: int = 4) -> List[Dict]:
#     """Generate MCQs for a topic using Groq."""
#     try:
#         results = vector_store.similarity_search(f"Topic: {topic}", k=1)
#         if not results:
#             logger.warning(f"No content found for topic {topic}")
#             return []
        
#         content = results[0].page_content
#         prompt = f"""
#         Generate {num_questions} multiple-choice questions for the topic '{topic}' based solely on the provided content, suitable for a {EDUCATION_LEVEL} student. Each question must have:
#         - A clear question
#         - 4 answer options (labeled A, B, C, D)
#         - The correct answer (as a letter: A, B, C, or D)
#         - A brief explanation for the correct answer
#         Return the questions in valid JSON format. Do not use the example content in the output.

#         Content: {content[:1500]}

#         JSON format:
#         [
#             {{
#                 "question": "Question text",
#                 "options": {{
#                     "A": "Option A",
#                     "B": "Option B",
#                     "C": "Option C",
#                     "D": "Option D"
#                 }},
#                 "correct_answer": "A",
#                 "explanation": "Explanation text"
#             }}
#         ]
#         """
#         response = llm.invoke(prompt)
#         logger.info(f"Raw Groq response for MCQs: {response.content.strip()}")
#         try:
#             mcqs = json.loads(response.content.strip())
#             if not isinstance(mcqs, list):
#                 raise ValueError("MCQs must be a list")
#             return mcqs
#         except json.JSONDecodeError as je:
#             logger.error(f"JSON parsing error: {je}")
#             return []
#         except ValueError as ve:
#             logger.error(f"Invalid MCQ format: {ve}")
#             return []
#     except Exception as e:
#         logger.error(f"Failed to generate MCQs for {topic}: {e}")
#         return []

# def store_mcq_performance(topic: str, score: float, answers: List[Dict]) -> None:
#     """Store MCQ performance in ChromaDB."""
#     try:
#         doc_id = str(uuid.uuid4())
#         document = Document(
#             page_content=f"Topic: {topic}\nScore: {score}\nAnswers: {json.dumps(answers)}",
#             metadata={"type": "mcq_performance", "topic": topic, "timestamp": time.time(), "score": score},
#             id=doc_id
#         )
#         vector_store.add_documents([document])
#         logger.info(f"Stored MCQ performance for topic {topic}")
#     except Exception as e:
#         logger.error(f"Failed to store MCQ performance: {e}")

# # Streamlit app
# st.title("Academic Explainer")

# # Sidebar for option selection
# option = st.sidebar.selectbox(
#     "Choose an option:",
#     ["Topic Description & YouTube Link", "YouTube Transcript & Summary", "MCQ Practice"]
# )

# if option == "Topic Description & YouTube Link":
#     st.header("Topic Description & YouTube Link")
#     syllabus_input = st.text_input("Enter syllabus topics (comma-separated, e.g., Photosynthesis, Quick Sort):")
    
#     if st.button("Process Topics"):
#         if syllabus_input:
#             topics = [topic.strip() for topic in syllabus_input.split(",")]
#             with st.spinner("Processing topics..."):
#                 results = process_syllabus(topics)
#                 for result in results:
#                     st.subheader(f"Topic: {result['topic'].capitalize()}")
#                     st.write("**Explanation:**")
#                     st.write(result['explanation'])
#                     st.write("**YouTube Link:**")
#                     st.write(result['video_url'])
#                     st.write("**Video Title:**")
#                     st.write(result['video_title'])
#                     st.markdown("---")
#         else:
#             st.warning("Please enter at least one topic.")

# elif option == "YouTube Transcript & Summary":
#     st.header("YouTube Transcript & Summary")
#     video_url = st.text_input("Enter a YouTube video URL:")
#     title = st.text_input("Enter a title for the video (optional):", value="Unknown")
    
#     if st.button("Process Video"):
#         if video_url:
#             with st.spinner("Processing video..."):
#                 result = process_youtube_video(video_url, title)
#                 st.subheader("Video Details")
#                 st.write("**Video URL:**")
#                 st.write(result['video_url'])
#                 st.write("**Transcript (excerpt):**")
#                 transcript = result['transcript']
#                 st.write(transcript[:500] + '...' if len(transcript) > 500 else transcript)
#                 st.write("**Summary:**")
#                 st.write(result['summary'])
#                 st.write(f"**Stored in vector DB:** {'✅' if result['stored'] else '❌'}")
                
#                 st.subheader("Ask a Question About the Video")
#                 query = st.text_input("Enter your question (or leave blank to skip):")
#                 if query and st.button("Submit Query"):
#                     with st.spinner("Searching database..."):
#                         responses = query_vector_db(query)
#                         if responses:
#                             st.subheader("Query Results")
#                             for resp in responses:
#                                 st.write("**Video URL:**")
#                                 st.write(resp['video_url'])
#                                 st.write("**Title:**")
#                                 st.write(resp['title'])
#                                 st.write("**Transcript (excerpt):**")
#                                 st.write(resp['transcript'][:500] + '...' if len(resp['transcript']) > 500 else resp['transcript'])
#                                 st.write("**Summary:**")
#                                 st.write(resp['summary'])
#                                 st.write(f"**Relevance Score:** {resp['score']:.2f}")
#                                 st.markdown("---")
#                         else:
#                             st.info("No relevant information found in the database.")
#         else:
#             st.warning("Please enter a valid YouTube URL.")

# elif option == "MCQ Practice":
#     st.header("MCQ Practice")
#     topics = fetch_topic_history()
    
#     if not topics:
#         st.warning("No topics found in history. Please process some topics or videos first.")
#     else:
#         selected_topic = st.selectbox("Select a topic to practice MCQs:", topics)
        
#         if st.button("Generate MCQs"):
#             with st.spinner("Generating MCQs..."):
#                 mcqs = generate_mcqs(selected_topic)
#                 if not mcqs:
#                     st.error("Failed to generate MCQs. Please try another topic or ensure content is stored.")
#                 else:
#                     st.session_state.mcqs = mcqs
#                     st.session_state.current_question = 0
#                     st.session_state.user_answers = []
#                     st.session_state.score = 0
                
#         if "mcqs" in st.session_state and st.session_state.mcqs:
#             current_q = st.session_state.current_question
#             if current_q < len(st.session_state.mcqs):
#                 question = st.session_state.mcqs[current_q]
#                 st.subheader(f"Question {current_q + 1} of {len(st.session_state.mcqs)}")
#                 st.write(question["question"])
                
#                 with st.form(key=f"question_{current_q}"):
#                     answer = st.radio("Select an answer:", list(question["options"].values()), key=f"answer_{current_q}")
#                     submit = st.form_submit_button("Submit Answer")
                    
#                     if submit:
#                         selected_option = [k for k, v in question["options"].items() if v == answer][0]
#                         is_correct = selected_option == question["correct_answer"]
#                         st.session_state.user_answers.append({
#                             "question": question["question"],
#                             "selected": selected_option,
#                             "correct": question["correct_answer"],
#                             "is_correct": is_correct,
#                             "explanation": question["explanation"]
#                         })
#                         if is_correct:
#                             st.session_state.score += 1
                        
#                         st.write(f"{'Correct!' if is_correct else 'Incorrect.'}")
#                         st.write(f"Explanation: {question['explanation']}")
#                         st.session_state.current_question += 1
#                         st.form_submit_button("Next Question", on_click=lambda: None)
            
#             else:
#                 score = st.session_state.score
#                 total = len(st.session_state.mcqs)
#                 st.subheader("Quiz Completed!")
#                 st.write(f"Your Score: {score}/{total} ({(score/total)*100:.1f}%)")
                
#                 store_mcq_performance(selected_topic, score/total, st.session_state.user_answers)
                
#                 for i, ans in enumerate(st.session_state.user_answers):
#                     st.write(f"**Question {i+1}:** {ans['question']}")
#                     st.write(f"Your Answer: {ans['selected']} ({'Correct' if ans['is_correct'] else 'Incorrect'})")
#                     st.write(f"Correct Answer: {ans['correct']}")
#                     st.write(f"Explanation: {ans['explanation']}")
#                     st.markdown("---")
                
#                 if st.button("Restart Quiz"):
#                     st.session_state.mcqs = []
#                     st.session_state.current_question = 0
#                     st.session_state.user_answers = []
#                     st.session_state.score = 0

# st.sidebar.markdown("---")
# st.sidebar.info("Built with Groq, Wikipedia, DuckDuckGo, and ChromaDB.")