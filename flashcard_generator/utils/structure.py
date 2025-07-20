
from typing import List, Dict, Any
from langchain_core.pydantic_v1 import BaseModel, Field

class Flashcard(BaseModel):
    question: str = Field(description="The question for the flashcard.")
    answer: str = Field(description="The answer to the flashcard question.")

class StudyGuide(BaseModel):
    mind_map: Dict[str, Any] = Field(description="A deeply nested dictionary representing a pure topic-subtopic hierarchy. Keys are topics, and values are always another nested dictionary of sub-topics.")
    flashcards: List[Flashcard] = Field(description="A list of 10 flashcards, each with a question and an answer.")