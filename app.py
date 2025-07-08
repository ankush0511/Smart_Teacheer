import streamlit as st
from llm import llm
from mcq import generate_mcqs, store_mcq_performance
from processes import process_syllabus, process_youtube_video
from YT_transcipt import query_vector_db,process_video
from history import fetch_topic_history, disambiguate_topic
from embedding import vector_store
from fetch_data import fetch_wikipedia_explanation, fetch_duckduckgo_explanation, fetch_youtube_video
from logger import logger
from pdf_maker import create_download_link,clean_text_for_pdf,generate_pdf_from_json
import json

# Initialize session state to persist results
if 'results' not in st.session_state:
    st.session_state.results = None




st.title("Academic Explainer")

# Sidebar for option selection
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Topic Description & YouTube Link", "YouTube Transcript & Summary", "MCQ Practice"]
)

if option == "Topic Description & YouTube Link":
    st.header("Topic Description & YouTube Link")
    syllabus_input = st.text_input("Enter syllabus topics (comma-separated, e.g., Photosynthesis, Quick Sort):")
    
    if st.button("Process Topics"):
        if syllabus_input:
            topics = [topic.strip() for topic in syllabus_input.split(",")]
            with st.spinner("Processing topics..."):
                # Process topics and store results in session state
                st.session_state.results = process_syllabus(topics)

                # Save results to JSON file
                with open("syllabus_results.json", "w", encoding="utf-8") as f:
                    json.dump(st.session_state.results, f, indent=4, ensure_ascii=False)
        else:
            st.warning("Please enter at least one topic.")

    # Display results if available in session state
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

    # Separate PDF generation section
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
 

elif option == "YouTube Transcript & Summary":
    st.header("YouTube Transcript & Summary")
    video_url = st.text_input("Enter a YouTube video URL:")
    title = st.text_input("Enter a title for the video (optional):", value="Unknown")
    
    if st.button("Process Video"):
        if video_url:
            with st.spinner("Processing video..."):
                result = process_youtube_video(video_url, title)
                st.subheader("Video Details")
                st.write("**Video URL:**")
                st.write(result['video_url'])
                st.write("**Transcript (excerpt):**")
                transcript = result['transcript']
                st.write(transcript[:500] + '...' if len(transcript) > 500 else transcript)
                st.write("**Summary:**")
                st.write(result['summary'])
                st.write(f"**Stored in vector DB:** {'✅' if result['stored'] else '❌'}")
                
                st.subheader("Ask a Question About the Video")
                query = st.text_input("Enter your question (or leave blank to skip):")
                if query and st.button("Submit Query"):
                    with st.spinner("Searching database..."):
                        responses = query_vector_db(query)
                        if responses:
                            st.subheader("Query Results")
                            for resp in responses:
                                st.write("**Video URL:**")
                                st.write(resp['video_url'])
                                st.write("**Title:**")
                                st.write(resp['title'])
                                st.write("**Transcript (excerpt):**")
                                st.write(resp['transcript'][:500] + '...' if len(resp['transcript']) > 500 else resp['transcript'])
                                st.write("**Summary:**")
                                st.write(resp['summary'])
                                st.write(f"**Relevance Score:** {resp['score']:.2f}")
                                st.markdown("---")
                        else:
                            st.info("No relevant information found in the database.")
        else:
            st.warning("Please enter a valid YouTube URL.")

elif option == "MCQ Practice":
    st.header("MCQ Practice")
    topics = fetch_topic_history()
    
    if not topics:
        st.warning("No topics found in history. Please process some topics or videos first.")
    else:
        selected_topic = st.selectbox("Select a topic to practice MCQs:", topics)
        # noMcq=st.text_input(int(input("enter the no of question")))
        num_questions = st.number_input("How many questions?", min_value=1, max_value=20)
        if st.button("Generate MCQs"):
            with st.spinner("Generating MCQs..."):
                mcqs = generate_mcqs(selected_topic,num_questions)
                if not mcqs:
                    st.error("Failed to generate MCQs. Please try another topic or ensure content is stored.")
                else:
                    st.session_state.mcqs = mcqs
                    st.session_state.current_question = 0
                    st.session_state.user_answers = []
                    st.session_state.score = 0
                
        if "mcqs" in st.session_state and st.session_state.mcqs:
            current_q = st.session_state.current_question
            if current_q < len(st.session_state.mcqs):
                question = st.session_state.mcqs[current_q]
                st.subheader(f"Question {current_q + 1} of {len(st.session_state.mcqs)}")
                st.write(question["question"])
                
                with st.form(key=f"question_{current_q}"):
                    answer = st.radio("Select an answer:", list(question["options"].values()), key=f"answer_{current_q}")
                    submit = st.form_submit_button("Submit Answer")
                    
                    if submit:
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
                        st.session_state.current_question += 1
                        st.form_submit_button("Next Question", on_click=lambda: None)
            
            else:
                score = st.session_state.score
                total = len(st.session_state.mcqs)
                st.subheader("Quiz Completed!")
                st.write(f"Your Score: {score}/{total} ({(score/total)*100:.1f}%)")
                
                store_mcq_performance(selected_topic, score/total, st.session_state.user_answers)
                
                for i, ans in enumerate(st.session_state.user_answers):
                    st.write(f"**Question {i+1}:** {ans['question']}")
                    st.write(f"Your Answer: {ans['selected']} ({'Correct' if ans['is_correct'] else 'Incorrect'})")
                    st.write(f"Correct Answer: {ans['correct']}")
                    st.write(f"Explanation: {ans['explanation']}")
                    st.markdown("---")
                
                if st.button("Restart Quiz"):
                    st.session_state.mcqs = []
                    st.session_state.current_question = 0
                    st.session_state.user_answers = []
                    st.session_state.score = 0

st.sidebar.markdown("---")
st.sidebar.info("Built with Groq, Wikipedia, DuckDuckGo, and ChromaDB.")