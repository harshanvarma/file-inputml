import streamlit as st
import base64
import fitz  # PyMuPDF
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class PDFAnalyzer:
    def __init__(self):
        """Initialize the PDFAnalyzer with the predefined API key."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_text(self, pdf_file):
        """Extract text from a PDF file."""
        text = ""
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text("text")
            pdf_document.close()
            return text
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def analyze_text(self, text, prompt="Analyze the patient's health data to identify deficiencies. Provide recommendations in the following format:\n\nRecommendations:\n- Food Item: [Name of the food item]\n  - Suggested Duration: [Duration for consumption]\n\nList each deficiency and specific food items that should be consumed to address it, along with the recommended duration for consumption."):
        """Analyze extracted text using OpenAI's API."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Changed from gpt-4o to gpt-4
                messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing PDF: {str(e)}"

def main():
    st.set_page_config(page_title="PDF Health Analysis", page_icon="🏥")
    
    st.title("PDF Health Analysis")
    st.write("Upload a PDF file for health analysis")

    # Initialize analyzer
    analyzer = PDFAnalyzer()

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        # Extract text
        st.info("Extracting text from PDF...")
        text = analyzer.extract_text(uploaded_file)
        
        if text.startswith("Error"):
            st.error(text)
        else:
            st.success("Text extracted successfully!")

            # Analyze text
            st.info("Analyzing PDF content...")
            result = analyzer.analyze_text(text)
            
            # Display results
            st.subheader("Analysis Result:")
            st.write(result)

            # Add download button for results
            result_str = f"""PDF Analysis Results

{result}
"""
            b64 = base64.b64encode(result_str.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="analysis_results.txt">Download Analysis Results</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()