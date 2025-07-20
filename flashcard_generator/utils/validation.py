from langchain_groq import ChatGroq
from flashcard_generator.utils.structure import ValidationResponse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def validate_input_content(content: str, groq_api_key: str) -> ValidationResponse:
    """
    NEW FUNCTION: Validates if the user input is a real topic using an LLM call.
    """
    try:
        model = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=groq_api_key)
        parser = JsonOutputParser(pydantic_object=ValidationResponse)

        prompt = PromptTemplate(
            template="""
            You are an intelligent validation assistant. Your task is to determine if the provided text is a real, understandable topic suitable for creating study materials.
            Topics can be academic subjects, concepts, book titles, etc.
            Invalid input would be random characters, gibberish, or nonsensical phrases.

            Analyze the following text and respond in the requested JSON format.

            TEXT:
            {content}

            FORMAT INSTRUCTIONS:
            {format_instructions}
            """,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | model | parser
        response = chain.invoke({"content": content})
        return response
    
    except Exception as e:
        print(f"Validation chain failed: {e}")
        return {"is_valid_topic": True, "reason": "Validation process encountered an error."}