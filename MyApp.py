import os
import argparse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document


class DocQA:
    """A class-based system to process legal documents and perform question-answering."""
    
    def __init__(self, file_path, question):
        self.file_path = file_path
        self.question = question
        self.vectorstore = None
        self.qa_chain = None
        self.documents = None

        # Load API key for Google Generative AI (Gemini)
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Please add it to the .env file.")
    
    def load_pdf(self):
        """Extracts text from the PDF file."""
        print("Loading PDF...")
        reader = PdfReader(self.file_path)
        self.documents = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Ignore empty pages
                metadata = {"source": self.file_path, "page": page_num + 1}
                self.documents.append({"text": text.strip(), "metadata": metadata})

    def create_vectorstore(self):
        """Creates a FAISS vectorstore with proper metadata."""
        print("Creating vectorstore...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = []

        for doc in self.documents:
            chunks = text_splitter.split_text(doc["text"])
            for chunk in chunks:
                # Create LangChain Document objects for compatibility
                split_docs.append(Document(page_content=chunk, metadata=doc["metadata"]))

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        self.vectorstore.save_local("faisss_index")


    def setup_qa_chain(self):
        """Sets up the question-answering chain using Google Generative AI."""

        print("Setting up the QA chain...")

        prompt_template = """
        You are a legal assistant. Use the context below to answer the question.

        Context:
        {context}?

        Question:
        {question}

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    def answer_question(self):
        """Answers the user's question with source citations."""
        self.load_pdf()
        self.create_vectorstore()
        self.setup_qa_chain()
        print("Answering the question...")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faisss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(self.question, k=6)

        # Combine the retrieved documents' content for context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        result = self.qa_chain(
            {
                "input_documents": docs,
                "question": self.question,
                "context": context
            },
            return_only_outputs=True
        )

        answer = result.get("output_text", "No answer found.")

        # Extract sources from retrieved documents' metadata
        sources = "\n".join(
            [f"- Page {doc.metadata.get('page', 'Unknown')}" for doc in docs]
        )

        print("\nAnswer:")
        print(answer)
        print("\nSources:")
        print(sources)
        return answer, sources


def main():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Legal Document Question-Answering System")
    parser.add_argument("file_path", type=str, help="Path to the PDF document")
    parser.add_argument("question", type=str, help="The question to ask about the document")
    args = parser.parse_args()

    # Run the QA system
    system = DocQA(file_path=args.file_path, question=args.question)
    system.answer_question()


if __name__ == "__main__":
    main()