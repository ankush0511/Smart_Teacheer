import streamlit as st
import json
import logging
from typing import Dict
import os
from langchain_groq import ChatGroq
from Copilot_MCQ.processes import process_syllabus
from Copilot_MCQ.history import fetch_topic_history
from Copilot_MCQ.embedding import vector_store
from Copilot_MCQ.pdf_maker import create_download_link, generate_pdf_from_json
from Copilot_MCQ.mcq import generate_mcqs, store_mcq_performance
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'mcqs' not in st.session_state:
    st.session_state.mcqs = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'last_submitted' not in st.session_state:
    st.session_state.last_submitted = None

st.title("Academic Copilot")

# Sidebar for option selection
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Academic Copilot", "MCQ Practice"]
)

if option == "Academic Copilot":
    st.header("Academic Copilot")
    syllabus_input = st.text_input("Enter syllabus topics (comma-separated, e.g., Photosynthesis, Quick Sort):")
    
    if st.button("Process Topics"):
        if syllabus_input:
            topics = [topic.strip() for topic in syllabus_input.split(",")]
            with st.spinner("Processing topics..."):
                st.session_state.results = process_syllabus(topics)
                with open("syllabus_results.json", "w", encoding="utf-8") as f:
                    json.dump(st.session_state.results, f, indent=4, ensure_ascii=False)
        else:
            st.warning("Please enter at least one topic.")

    if st.session_state.results:
        for result in st.session_state.results:
            st.subheader(f"Topic: {result['topic'].capitalize()}")
            st.write("**Explanation:**")
            st.write(result['explanation'])
            st.write("**YouTube Link:**")
            st.write(result['video_url'])
            st.write("**Video Title:**")
            st.write(result['video_title'])
            st.markdown("---")

    st.title("📄 Export Syllabus Explanation as PDF")
    if st.button("Generate PDF"):
        if st.session_state.results:
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_pdf_from_json(st.session_state.results)
                html_link = create_download_link(pdf_bytes, "syllabus_explanations")
                st.success("✅ PDF generated successfully!")
                st.markdown(html_link, unsafe_allow_html=True)
        else:
            st.error("❌ No results available. Please process topics first.")

elif option == "MCQ Practice":
    st.header("MCQ Practice")
    topics = fetch_topic_history()
    
    if not topics:
        st.warning("No topics found in history. Please process some topics or videos first.")
    else:
        selected_topic = st.selectbox("Select a topic to practice MCQs:", topics, key="topic_select")
        num_questions = st.number_input("How many questions?", min_value=1, max_value=20, value=5, key="num_questions")
        
        if st.button("Start Quiz", key="start_quiz"):
            with st.spinner("Generating MCQs..."):
                st.session_state.mcqs = generate_mcqs(selected_topic, num_questions)
                st.session_state.current_question = 0
                st.session_state.user_answers = []
                st.session_state.score = 0
                st.session_state.quiz_started = True
                st.session_state.last_submitted = None
                if not st.session_state.mcqs:
                    st.error("Failed to generate MCQs. Please try another topic or ensure content is stored.")
                    st.session_state.quiz_started = False

        if st.session_state.quiz_started and st.session_state.mcqs:
            current_q = st.session_state.current_question
            if current_q < len(st.session_state.mcqs):
                question = st.session_state.mcqs[current_q]
                st.subheader(f"Question {current_q + 1} of {len(st.session_state.mcqs)}")
                st.write(question["question"])
                
                # Unique form key with timestamp to prevent resubmission issues
                form_key = f"question_{current_q}_{st.session_state.get('form_counter', 0)}"
                with st.form(key=form_key):
                    answer = st.radio("Select an answer:", list(question["options"].values()), key=f"answer_{current_q}")
                    submit = st.form_submit_button("Submit Answer")
                    
                    if submit and st.session_state.last_submitted != form_key:
                        selected_option = [k for k, v in question["options"].items() if v == answer][0]
                        is_correct = selected_option == question["correct_answer"]
                        st.session_state.user_answers.append({
                            "question": question["question"],
                            "selected": selected_option,
                            "correct": question["correct_answer"],
                            "is_correct": is_correct,
                            "explanation": question["explanation"]
                        })
                        if is_correct:
                            st.session_state.score += 1
                        
                        st.write(f"{'Correct!' if is_correct else 'Incorrect.'}")
                        st.write(f"Explanation: {question['explanation']}")
                        st.session_state.last_submitted = form_key
                        st.session_state.current_question += 1
                        # Increment form counter to ensure unique form key on next question
                        st.session_state.form_counter = st.session_state.get('form_counter', 0) + 1

            else:
                st.session_state.quiz_started = False
                score = st.session_state.score
                total = len(st.session_state.mcqs)
                st.subheader("Quiz Completed!")
                st.write(f"Your Score: {score}/{total} ({(score/total)*100:.1f}%)")
                
                store_mcq_performance(selected_topic, score/total, st.session_state.user_answers)
                
                for i, ans in enumerate(st.session_state.user_answers):
                    st.write(f"**Question {i+1}:** {ans['question']}")
                    st.write(f"Your Answer: {ans['selected']} ({'Correct' if ans['is_correct'] else 'Incorrect'})")
                    st.write(f"Explanation: {ans['explanation']}")
                    st.markdown("---")
                
                if st.button("Restart Quiz", key="restart_quiz"):
                    st.session_state.mcqs = []
                    st.session_state.current_question = 0
                    st.session_state.user_answers = []
                    st.session_state.score = 0
                    st.session_state.quiz_started = False
                    st.session_state.last_submitted = None

st.sidebar.markdown("---")
st.sidebar.info("Built with Groq, Wikipedia, DuckDuckGo, and ChromaDB.")