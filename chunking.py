import requests
import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Define the reference schema
class Reference(BaseModel):
    number: int
    author_name: Optional[str] = Field(None, description="Author's name")
    title: Optional[str] = Field(None, description="Title of the paper")
    publication: Optional[str] = Field(None, description="Publication source")
    date_published: Optional[str] = Field(None, description="Date of publication")
    arxiv_id: Optional[str] = Field(None, description="ArXiv ID if available")

class ReferenceList(BaseModel):
    references: List[Reference]

# Function to download the arXiv paper
def download_arxiv_pdf(arxiv_id: str) -> str:
    save_path = f"{arxiv_id}.pdf"
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    else:
        raise Exception("Failed to download paper")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from the PDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    return full_text

# Function to split text into chunks
def split_text_into_chunks(text: str) -> List[str]:
    """Splits text into chunks of 1024 characters with an overlap of 100."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)

# Function to extract only the References section
def extract_references_section(pdf_path: str) -> str:
    """Extract only the References section from the PDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    
    refs_start = full_text.lower().find("references")
    if refs_start != -1:
        return full_text[refs_start:]  # Extract from "References" onward
    return full_text  # Fallback to full text if "References" not found

# Function to clean the arXiv ID
def clean_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
    if arxiv_id:
        match = re.search(r'(\d+\.\d+)', arxiv_id)
        return match.group() if match else None
    return None

# Function to extract references using LLM
def extract_references(text: str, groq_api_key: str) -> ReferenceList:
    model = ChatGroq(model_name="llama-3.2-90b-vision-preview", groq_api_key=groq_api_key)
    parser = PydanticOutputParser(pydantic_object=ReferenceList)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract references from the provided research paper text, ensuring that all fields are included whenever possible. Format them as JSON."),
        ("human", "Text:\n{text}\n\n{format_instructions}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | model | parser
    references = chain.invoke({"text": text})
    
    for ref in references.references:
        ref.arxiv_id = clean_arxiv_id(ref.arxiv_id)
    
    return references

# Function to convert written numbers to digits
def words_to_number(word: str) -> Optional[int]:
    try:
        doc = nlp(word)
        for token in doc:
            if token.like_num:
                return int(token.text)
        return None
    except ValueError:
        return None

# Function to retrieve a specific reference by number
def get_reference_by_number(references: ReferenceList, query: str) -> Optional[Reference]:
    """Extract the reference number using NLP."""
    doc = nlp(query)
    number = None
    
    for token in doc:
        if token.like_num:
            try:
                number = int(token.text)
                break
            except ValueError:
                continue
        elif token.text.lower().endswith("th") or token.text.lower().endswith("st") or token.text.lower().endswith("nd") or token.text.lower().endswith("rd"):
            try:
                number = int(token.text[:-2])
                break
            except ValueError:
                continue
        else:
            num = words_to_number(token.text)
            if num is not None:
                number = num
                break
    
    if number is not None:
        for ref in references.references:
            if ref.number == number:
                return ref
    return None

# Main execution function
def main(arxiv_id: str):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in the .env file.")
    
    pdf_path = download_arxiv_pdf(arxiv_id)
    text = extract_references_section(pdf_path)
    references = extract_references(text, groq_api_key)
    
    print("Extracted References:")
    print(json.dumps(references.dict(), indent=4))
    
    query = input("Enter the reference number you want to know more about: ")
    reference = get_reference_by_number(references, query)
    
    if reference:
        print(json.dumps(reference.dict(), indent=4))
        if reference.arxiv_id:
            print(f"Downloading referenced paper with ArXiv ID: {reference.arxiv_id}")
            ref_pdf_path = download_arxiv_pdf(reference.arxiv_id)
            print("Download complete.")
            
            extracted_text = extract_text_from_pdf(ref_pdf_path)
            text_chunks = split_text_into_chunks(extracted_text)
            print("Extracted and chunked text:", text_chunks)
    else:
        print("Reference not found.")

# Example usage
if __name__ == "__main__":
    arxiv_id = "1206.1106"  # Example arXiv ID
    main(arxiv_id)
