# Import necessary libraries
import streamlit as st  # Streamlit is used to create a user interface
import requests  # For making HTTP requests
from bs4 import BeautifulSoup  # For parsing HTML content of a web page
import nltk  # Natural Language Toolkit for text processing
from transformers import BertTokenizer, BertForQuestionAnswering  # Hugging Face Transformers for BERT
import torch  # PyTorch for computation
import csv  # For CSV file manipulation
import io
import fitz  # PyMuPDF for extracting text from PDF files
from summarizer import Summarizer  # For text summarization

# Download necessary resources for NLTK and BERT
nltk.download('punkt')

# Function to extract the answer using the "question_answering" model
def get_answer(question, context):
    # Load the pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Preprocess the data to feed into the BERT model
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Find the start and end indices of the answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Convert the indices to text
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# Main function
def main():
    # Streamlit application title
    st.title("Data Extraction by ChatBot")

    # Input field for the article link
    link = st.text_input("Enter the article link:")
    extract_button = st.button("Extract Text and Generate Summary")

    # If the "Extract" button is clicked
    if extract_button:
        if link:
            if link.lower().endswith(".pdf"):  # If the link points to a PDF
                response = requests.get(link)
                if response.status_code == 200:
                    pdf_stream = io.BytesIO(response.content)

                    # Open the PDF with PyMuPDF
                    pdf_file = fitz.open(stream=pdf_stream, filetype="pdf")
                    article_text = ""
                    for page_num in range(pdf_file.page_count):
                        page = pdf_file.load_page(page_num)
                        page_text = page.get_text()
                        article_text += page_text

                    # Limit the length of extracted text to meet BERT constraints
                    max_length = 5120
                    if len(article_text) > max_length:
                        article_text = article_text[:max_length]

                    # Store the extracted text in the Streamlit application session
                    st.session_state.article_text = article_text
                    st.success("Text extracted successfully!")

                    # Generate a summary of the extracted text
                    summarizer = Summarizer()
                    summary = summarizer(article_text)
                    st.subheader("Generated Summary:")
                    st.write(summary)
            else:  # If the link points to a web page
                response = requests.get(link)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Extract text from all paragraphs on the web page
                    article_text = ' '.join([p.get_text() for p in soup.find_all('p')])

                    # Store the extracted text in the Streamlit application session
                    st.session_state.article_text = article_text
                    st.success("Text extracted successfully!")

                    # Create a CSV file to store the link and extracted text
                    article_filename = link.replace("/", "").replace(":", "") + ".csv"
                    with open(article_filename, mode="w", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file, delimiter=";")
                        writer.writerow(["Link", "Extracted Text"])
                        writer.writerow([link, article_text])

    # If the article text has been successfully extracted
    if "article_text" in st.session_state:
        # Display the extracted text from the article
        st.header("Extracted Text from the Article:")
        st.text_area("", value=st.session_state.article_text, height=300)
        
        # Generate a summary of the extracted text
        summarizer = Summarizer()
        summary = summarizer(st.session_state.article_text)
        st.subheader("Generated Summary:")
        st.write(summary)

        # Input field for the user's question
        user_input = st.text_input("Ask a question:")

        # Button to get the answer to the question
        if st.button("Get Answer"):
            answer = get_answer(user_input, st.session_state.article_text)
            if answer.strip() == "" or answer == "[CLS]":
                answer = "The answer does not exist in this article."
            st.subheader("Answer:")
            st.write(answer)

            # Add the question and answer to the CSV file
            article_filename = link.replace("/", "").replace(":", "") + ".csv"
            with open(article_filename, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow([user_input, answer])

# Script entry point
if __name__ == "__main__":
    main()
