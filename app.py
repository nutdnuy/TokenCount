import streamlit as st
import pdfplumber
import tiktoken

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from each page of a PDF file."""
    full_text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                full_text += f'\n--- Page {page_number} ---\n'
                full_text += page_text + '\n'
    return full_text

# Function to count tokens using tiktoken
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in the given string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to calculate cost for different models
def calculate_costs(token_count: int) -> dict:
    """Calculate the cost for different models based on the token count."""
    cost_per_1m_tokens = {
        'GPT-4': 2.50,            # $2.50 per 1M tokens
        'GPT-4 Mini': 0.150,      # $0.15 per 1M tokens
        'O1-Preview': 15.00,      # $15.00 per 1M tokens
        'GPT-4 Turbo': 10.00      # $10.00 per 1M tokens
    }
    costs = {model: (token_count / 1_000_000) * price for model, price in cost_per_1m_tokens.items()}
    return costs

# Title of the app
st.title("📄 PDF to Text Extractor with Token Count and Cost Estimation")

# Upload PDF file
uploaded_pdf = st.file_uploader("Upload your PDF file here", type=["pdf"])

if uploaded_pdf:
    st.subheader("📋 Extracted Text from PDF")
    
    # Extract the text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_pdf)
    
    if extracted_text:
        # Count the tokens in the extracted text
        token_count = num_tokens_from_string(extracted_text)
        
        # Display the extracted text
        #st.text_area("Extracted Text", extracted_text, height=400)
        
        # Display token count
        st.metric(label="📏 Total Tokens", value=f"{token_count} tokens")
        
        # Calculate the costs for each model
        costs = calculate_costs(token_count)
        
        # Display the cost for each model
        st.subheader("💲 Cost Calculation for Token Usage")
        for model, cost in costs.items():
            st.metric(label=f"Cost for {model}", value=f"${cost:.4f}")
        
        # Download option for the extracted text
        st.download_button(
            label="💾 Download Extracted Text",
            data=extracted_text,
            file_name='extracted_text.txt',
            mime='text/plain'
        )
    else:
        st.warning("No text could be extracted from this PDF.")
