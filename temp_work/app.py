import streamlit as st
from main import process_syllabus, process_youtube_video, query_vector_db

st.title("Academic Explainer")
option = st.selectbox("Choose an option", ["Topic Description & YouTube Link", "YouTube Transcript & Summary", "Exit"])

if option == "Topic Description & YouTube Link":
    syllabus = st.text_input("Enter syllabus topics (comma-separated):")
    if st.button("Process"):
        results = process_syllabus(syllabus.split(","))
        for result in results:
            st.subheader(result["topic"])
            st.write(f"**Explanation**: {result['explanation']}")
            st.write(f"**YouTube Link**: {result['video_url']}")

elif option == "YouTube Transcript & Summary":
    video_url = st.text_input("Enter YouTube video URL:")
    title = st.text_input("Enter video title (optional):", "Unknown")
    if st.button("Process"):
        result = process_youtube_video(video_url, title)
        st.write(f"**Video URL**: {result['video_url']}")
        st.write(f"**Transcript**: {result['transcript'][:500] + '...' if len(result['transcript']) > 500 else result['transcript']}")
        st.write(f"**Summary**: {result['summary']}")