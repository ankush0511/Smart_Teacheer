import pypdf
import streamlit as st
def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts text from an uploaded PDF file.
    """
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""