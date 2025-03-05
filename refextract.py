import arxiv
import requests
import json
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyDBG61t1v03utFkxNaixa_N2ZCVkVZZCLQ")

# Function to download the paper PDF from arXiv
def download_arxiv_paper(arxiv_id):
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    
    for result in client.results(search):
        pdf_url = result.pdf_url
        response = requests.get(pdf_url)
        if response.status_code == 200:
            pdf_path = f"{arxiv_id}.pdf"
            with open(pdf_path, "wb") as file:
                file.write(response.content)
            return pdf_path
    
    return None

# Function to extract references using Gemini 2.0 Flash API
def extract_references_from_paper(pdf_path):
    model = genai.GenerativeModel("gemini-2.0-flash")
    with open(pdf_path, "rb") as file:
        pdf_data = file.read()
    
    response = model.generate_content([
        {"mime_type": "application/pdf", "data": pdf_data},
        "Extract the references section from this academic paper and return them in structured JSON format."
    ])
    
    return response.text if response else json.dumps({"error": "Failed to extract references."})

# Main function
def get_arxiv_references(arxiv_id):
    pdf_path = download_arxiv_paper(arxiv_id)
    if not pdf_path:
        return json.dumps({"error": "Failed to download paper."})
    
    references_json = extract_references_from_paper(pdf_path)
    return references_json

# Example usage
arxiv_id = "2410.14062"  # Example arXiv ID
references = get_arxiv_references(arxiv_id)
print(references)
