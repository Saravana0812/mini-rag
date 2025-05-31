# Mini RAG System

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using Streamlit for a user interface. It allows you to chat with your documents.

## Setup

1.  **Clone this repository.**
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the environment:**
    *   Windows (Command Prompt): `venv\Scripts\activate`
    *   Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
    *   macOS/Linux: `source venv/bin/activate`
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Place your documents:** Put your `.txt` or `.pdf` files into the `documents/` folder.

## Running the App

1.  **Ensure your virtual environment is activated.**
2.  **Run the Streamlit app:**
    ```bash
    python -m streamlit run main.py
    ```
3.  Open the provided Local URL in your browser.

## Docker Deployment

The application can be containerized using Docker for consistent deployments, including to cloud platforms like Azure Web Apps.

To build the Docker image:
```bash
docker build -t mini-rag-system .
```
To run the Docker container locally:
```bash
docker run -p 8080:8080 mini-rag-system
```

## Usage

*   Type your question in the chat input.
*   The system will answer based on your documents.
*   Click "Clear Chat History" in the sidebar to reset the conversation.
*   To add new documents, use the "Upload a new document" feature in the sidebar. The `documents/` folder will be created automatically if it doesn't exist.
*   To re-index documents (after adding new ones or if the index is missing), click the "Update" button in the sidebar.
