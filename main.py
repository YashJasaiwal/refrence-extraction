import requests
import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import base64
import io

# Load environment variables from .env file
load_dotenv()

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
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    return full_text

# Function to split text into chunks
def split_text_into_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)

# Function to extract only the References section using regex
def extract_references_section(pdf_path: str) -> str:
    """Extract only the References section from the PDF using regex for better detection."""
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text("text") for page in doc])
    
    # Improved regex to match different variations of "References"
    match = re.search(r"(References|Bibliography|REFERENCES AND NOTES)(.*)", full_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(2).strip()  # Extract from the matched section onward
    return full_text  # Fallback to full text if section not found

# Function to clean the arXiv ID
def clean_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
    if arxiv_id:
        match = re.search(r'(\d+\.\d+)', arxiv_id)
        return match.group() if match else None
    return None

# Function to extract references using LLM (Improved Prompt & Gemini-1.5-Pro)
def extract_references(text: str, gemini_api_key: str) -> ReferenceList:
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    parser = PydanticOutputParser(pydantic_object=ReferenceList)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Extract ALL references from the provided research paper text, ensuring that:\n"
         "- Each reference has a valid number matching the paper.\n"
         "- Extract the author's name, title, publication, and date (if available).\n"
         "- If arXiv ID is present, extract it correctly.\n"
         "- Do NOT hallucinate extra references.\n"
         "- Return output in strict JSON format."),
        ("human", "Text:\n{text}\n\n{format_instructions}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser
    
    references = chain.invoke({"text": text})

    # Post-processing to clean and validate extracted references
    valid_references = []
    seen_numbers = set()
    
    for ref in references.references:
        ref.arxiv_id = clean_arxiv_id(ref.arxiv_id)
        
        # Remove duplicates and hallucinated numbers
        if ref.number not in seen_numbers and ref.number > 0:
            seen_numbers.add(ref.number)
            valid_references.append(ref)
    
    return ReferenceList(references=valid_references)

# Function to extract reference number using Gemini
def extract_reference_number(user_query: str, gemini_api_key: str) -> Optional[int]:
    """Uses Gemini LLM to extract a reference number from the user query."""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

    # Define a JSON schema for structured output
    parser = PydanticOutputParser(pydantic_object=Reference)

    # Prompt to enforce proper extraction
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are extracting the reference number from a user query."),
        ("human", "Extract the reference number from the following query and return only the number in JSON format.\n\nQuery: {query}\n\n{format_instructions}")
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    try:
        response = chain.invoke({"query": user_query})

        # Debugging: Print raw response
        print("Raw LLM Response:", response)

        if isinstance(response, dict) and "number" in response:
            return int(response["number"])
        elif isinstance(response, Reference):
            return response.number
        else:
            print("Unexpected response format from Gemini.")
            return None
    except Exception as e:
        print(f"Error extracting reference number: {e}")
        return None


# Function to retrieve a specific reference by number using Gemini
def get_reference_by_number(references: ReferenceList, user_query: str, gemini_api_key: str) -> Optional[Reference]:
    """Extract the reference number using Gemini AI."""
    reference_number = extract_reference_number(user_query, gemini_api_key)
    
    if reference_number is not None:
        for ref in references.references:
            if ref.number == reference_number:
                return ref
    return None

def extract_references_with_vision(pdf_path: str, gemini_api_key: str) -> ReferenceList:
    """Extract references from PDF using Gemini 2.0 Flash."""
    doc = fitz.open(pdf_path)
    
    # 1. Enhanced reference section detection
    full_text = ""
    found_references = False
    reference_pages = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        # Look for reference section markers
        if re.search(r'(?:^|\n)\s*(?:References?|Bibliography|REFERENCES|Citations)\s*(?:\n|$)', text, re.MULTILINE):
            found_references = True
            reference_pages.append(page_num)
            # Include next few pages as they likely contain references
            reference_pages.extend(range(page_num + 1, min(page_num + 5, doc.page_count)))
    
    if not found_references:
        print("No references section found.")
        return ReferenceList(references=[])
    
    # 2. Extract text from reference pages with context
    references_text = ""
    for page_num in sorted(set(reference_pages)):
        page = doc[page_num]
        text = page.get_text()
        # Clean but preserve structure
        text = re.sub(r'\f', '\n', text)  # Form feeds to newlines
        text = re.sub(r'(?<=[.!?])\s{2,}', '\n', text)  # Split on multiple spaces after sentences
        references_text += text + "\n"
    
    # 3. Process each reference individually
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_key=gemini_api_key,
        temperature=0.1,
        max_output_tokens=8000,
        top_p=0.95,
    )
    parser = PydanticOutputParser(pydantic_object=ReferenceList)
    
    # 4. Improved prompt for single reference processing
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert in extracting information from academic references. Given the text of the references section, identify and extract each reference. For each reference, extract:\n"
         "1. The reference number (if present)\n"
         "2. Author names (ALL authors, in order, full names)\n"
         "3. Paper title (complete, with original capitalization)\n"
         "4. Publication details (journal/conference/book)\n"
         "5. Publication date/year\n"
         "6. ArXiv ID if present (format: XXXX.XXXXX)\n\n"
         "Instructions:\n"
         "- Keep exact formatting and capitalization\n"
         "- Include ALL authors\n"
         "- Extract COMPLETE title\n"
         "- Do not invent or hallucinate missing information\n"
         "- Return valid JSON only"),
        ("human", "References text: {text}\n\n{format_instructions}")
    ]).partial(format_instructions=parser.get_format_instructions())

    all_references = []
    seen_numbers = set()
    
    try:
        chain = prompt | model | parser
        result = chain.invoke({"text": references_text})
        
        for ref in result.references:
            if ref.number not in seen_numbers and ref.number > 0:
                # Clean fields
                ref.author_name = ref.author_name.strip() if ref.author_name else None
                ref.title = ref.title.strip() if ref.title else None
                ref.publication = ref.publication.strip() if ref.publication else None
                ref.arxiv_id = clean_arxiv_id(ref.arxiv_id)
                
                seen_numbers.add(ref.number)
                all_references.append(ref)
                
    except Exception as e:
        print(f"Error processing references: {e}")
        return ReferenceList(references=[])
    
    # Verify completeness
    if all_references:
        max_ref = max(ref.number for ref in all_references)
        missing = set(range(1, max_ref + 1)) - set(ref.number for ref in all_references)
        if missing:
            print(f"Warning: Missing references: {missing}")
    
    return ReferenceList(references=sorted(all_references, key=lambda x: x.number))

# Update the main function to use vision-based extraction
def main(arxiv_id: str):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file.")
    
    pdf_path = download_arxiv_pdf(arxiv_id)
    references = extract_references_with_vision(pdf_path, gemini_api_key)
    
    print("Extracted References:")
    print(json.dumps(references.dict(), indent=4))
    
    query = input("Enter the reference number you want to know more about: ")
    reference = get_reference_by_number(references, query, gemini_api_key)
    
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
    arxiv_id = "2401.12491"  # Example ArXiv ID
    main(arxiv_id)
