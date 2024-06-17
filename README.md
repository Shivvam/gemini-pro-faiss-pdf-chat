
# DocInsights: Extract Answers from Documents

This application allows users to upload documents (PDF, DOCX, PPTX) and ask questions about the content. It extracts text from the uploaded files, processes the text to create embeddings, and then uses a conversational AI model to answer questions based on the document content.

## Features

- **Document Upload**: Users can upload PDF, DOCX, and PPTX files.
- **Text Extraction**: Extracts text from the uploaded documents.
- **Text Chunking**: Splits the extracted text into manageable chunks for processing.
- **Embedding Creation**: Uses Google Generative AI to create embeddings for the text chunks.
- **Question Answering**: Allows users to ask questions about the document content and receives detailed answers.

## Installation

1. Clone the repository.
2. Install the required packages using pip:
   
   ```
   pip install -r requirements.txt
   ```
3. Get the google api ket from here https://aistudio.google.com/app/apikey

4. Set up your Google API Key in a `.env` file:

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Run the application:
   
   ```
   streamlit run DocInsights.py
   ```
3. Open the provided URL in your web browser.
4. Upload your documents and click "Submit & Process".
5. Ask a question about the document content in the input field and click "Ask".

