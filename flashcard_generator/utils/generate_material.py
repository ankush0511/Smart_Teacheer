from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from flashcard_generator.utils.structure import StudyGuide

load_dotenv()

def generate_study_materials(content: str, groq_api_key: str) -> dict:
    """
    Generates a mind map and flashcards using LangChain and ChatGroq.
    """
    try:
        # 1. Initialize the ChatGroq model
        model = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192",
            api_key=groq_api_key
        )

        # 2. Set up the output parser
        parser = JsonOutputParser(pydantic_object=StudyGuide)

        # 3. Define the FINAL Prompt Template
        # This prompt strictly enforces a topics-only hierarchy for the mind map.
        prompt = PromptTemplate(
            template="""
            You are a world-class AI expert at creating deeply hierarchical outlines.
            Your task is to analyze the provided text and create a comprehensive study guide in a structured JSON format.

            The study guide must contain two main keys: "mind_map" and "flashcards".

            1.  **mind_map**:
                -   This must be a deeply nested JSON object representing a pure topic-subtopic hierarchy.
                -   **Keys** must always be the name of a topic or concept.
                -   **Values** must ALWAYS be another nested JSON object representing the sub-topics for that key.
                -   **CRITICAL RULE: Do NOT include any descriptions, definitions, or explanations as string values.** The mind map must ONLY contain the structure of topics. If a topic has no further sub-topics, its value should be an empty object {{}}.
                -   Your goal is to create the deepest and most exhaustive hierarchy possible, breaking down every concept from the source text into its constituent parts.

            2.  **flashcards**:
                -   This must be a list of 10 insightful flashcards based on the most important information in the text.
                -   Each flashcard should be a JSON object with a "question" and an "answer" field.

            Please process the following content and generate the detailed study guide.

            CONTENT:
            {content}

            FORMAT INSTRUCTIONS:
            {format_instructions}
            """,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # 4. Create the LangChain chain
        chain = prompt | model | parser

        # 5. Invoke the chain with the user's content
        response = chain.invoke({"content": content})
        return response

    except Exception as e:
        st.error(f"An error occurred during generation: {e}")
        st.warning("Failed to generate study materials. The model may have returned an invalid format. Please try again with a different input or a more specific topic.")
        return None