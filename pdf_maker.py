import re
import streamlit as st
from fpdf import FPDF
import base64

# Function to create download link for PDF
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val is bytes
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">📥 Download PDF</a>'

# Helper function to clean text for latin-1 encoding
def clean_text_for_pdf(text):
    # Replace common Unicode characters with latin-1 compatible ones
    replacements = {
        '\u2013': '-',  # en-dash to hyphen
        '\u2014': '-',  # em-dash to hyphen
        '\u2018': "'",  # left single quote to straight quote
        '\u2019': "'",  # right single quote to straight quote
        '\u201c': '"',  # left double quote to straight quote
        '\u201d': '"',  # right double quote to straight quote
        '\u2026': '...'  # ellipsis to three dots
    }
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    # Optionally, remove any remaining non-latin-1 characters
    return ''.join(c for c in text if ord(c) < 256 or c.isspace())

# Function to generate PDF from JSON data with bold formatting
def generate_pdf_from_json(json_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for item in json_data:
        topic = clean_text_for_pdf(item.get('topic', '').capitalize())
        explanation = clean_text_for_pdf(item.get('explanation', ''))

        # Render topic in bold
        pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 10, topic)
        pdf.ln(2)

        # Parse explanation for **bold** formatting
        pdf.set_font("Arial", '', 12)
        parts = re.split(r'\*\*(.*?)\*\*', explanation)
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Non-bold text
                pdf.set_font("Arial", '', 14)
            else:
                # Bold text (between ** markers)
                pdf.set_font("Arial", 'B', 12)
            if part:
                pdf.write(5, part)
        pdf.ln(5)  # Add spacing after explanation

    return pdf.output(dest="S").encode("latin-1")

