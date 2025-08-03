import json
import logging
import os
from dotenv import load_dotenv
from typing import Dict, List
from langchain_groq import ChatGroq
from processes import process_syllabus
from history import fetch_topic_history
from embedding import vector_store
from pdf_maker import generate_pdf_from_json
from mcq import generate_mcqs, store_mcq_performance

# Load environment variables
load_dotenv()

# Validate GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY is not set.")
    exit(1)

# Initialize Groq LLM
try:
    llm = ChatGroq(model_name="gemma2-9b-it", api_key=GROQ_API_KEY, temperature=0.7)
except Exception as e:
    print(f"Error: Failed to initialize Groq LLM: {e}")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_pdf(json_data, filename="syllabus_explanations.pdf"):
    """Save PDF from JSON data."""
    try:
        pdf_bytes = generate_pdf_from_json(json_data)
        with open(filename, "wb") as f:
            f.write(pdf_bytes)
        print(f"PDF saved as {filename}")
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}")
        print(f"Error: Failed to save PDF: {e}")

def run_academic_copilot():
    """Process syllabus topics and generate explanations and PDF."""
    print("\n=== Academic Copilot ===")
    syllabus_input = input("Enter syllabus topics (comma-separated, e.g., Photosynthesis, Quick Sort): ")
    if not syllabus_input:
        print("Error: Please enter at least one topic.")
        return
    
    topics = [topic.strip() for topic in syllabus_input.split(",")]
    print("Processing topics...")
    results = process_syllabus(topics)
    
    if results:
        with open("syllabus_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print("Syllabus results saved to syllabus_results.json")
        
        for result in results:
            print(f"\nTopic: {result['topic'].capitalize()}")
            print("Explanation:")
            print(result['explanation'])
            print("YouTube Link:")
            print(result['video_url'])
            print("Video Title:")
            print(result['video_title'])
            print("-" * 50)
        
        save_pdf(results)
    else:
        print("Error: No results generated for the provided topics.")

def run_mcq_practice():
    """Run MCQ practice session."""
    print("\n=== MCQ Practice ===")
    topics = fetch_topic_history()
    
    if not topics:
        print("Error: No topics found in history. Please process some topics first.")
        return
    
    print("Available topics:", ", ".join(topics))
    selected_topic = input("Select a topic to practice MCQs: ").strip().lower()
    if selected_topic not in topics:
        print("Error: Invalid topic selected.")
        return
    
    try:
        num_questions = int(input("How many questions? (1-20): "))
        if not 1 <= num_questions <= 20:
            raise ValueError("Number of questions must be between 1 and 20.")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("Generating MCQs...")
    mcqs = generate_mcqs(selected_topic, num_questions)
    if not mcqs:
        print("Error: Failed to generate MCQs. Please try another topic or ensure content is stored.")
        return
    
    score = 0
    user_answers = []
    for i, question in enumerate(mcqs, 1):
        print(f"\nQuestion {i} of {len(mcqs)}: {question['question']}")
        for key, option in question['options'].items():
            print(f"{key}: {option}")
        
        answer = input("Select an answer (A, B, C, D): ").strip().upper()
        if answer not in question['options']:
            print("Invalid answer. Skipping question.")
            continue
        
        is_correct = answer == question['correct_answer']
        user_answers.append({
            "question": question["question"],
            "selected": answer,
            "correct": question["correct_answer"],
            "is_correct": is_correct,
            "explanation": question["explanation"]
        })
        
        if is_correct:
            score += 1
            print("Correct!")
        else:
            print("Incorrect.")
        print(f"Explanation: {question['explanation']}")
    
    print(f"\nQuiz Completed! Your Score: {score}/{len(mcqs)} ({(score/len(mcqs))*100:.1f}%)")
    store_mcq_performance(selected_topic, score/len(mcqs), user_answers)
    
    print("\nReview Answers:")
    for i, ans in enumerate(user_answers, 1):
        print(f"Question {i}: {ans['question']}")
        print(f"Your Answer: {ans['selected']} ({'Correct' if ans['is_correct'] else 'Incorrect'})")
        print(f"Explanation: {ans['explanation']}")
        print("-" * 50)

def main():
    """Main function to run the Academic Copilot application."""
    print("Welcome to Academic Copilot")
    print("Built with Groq, Wikipedia, DuckDuckGo, and ChromaDB")
    print('Academic Copilot is a tool to help you learn and practice academic topics through explanations and MCQs.')
    run_academic_copilot()
if __name__ == "__main__":
    main()