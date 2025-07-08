        

import json
import streamlit as st
from fpdf import FPDF
import base64

# Function to create download link for PDF
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val is bytes
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">📥 Download PDF</a>'

# Function to generate PDF from JSON data
def generate_pdf_from_json(json_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for item in json_data:
        topic = item.get('topic', '').capitalize()
        explanation = item.get('explanation', '')

        pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 10, topic)
        
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, explanation)
        pdf.ln(5)

    return pdf.output(dest="S").encode("latin-1")