from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import re
from dotenv import load_dotenv
from urllib.parse import urlparse
import mimetypes
import requests
from bs4 import BeautifulSoup
import torch

load_dotenv()

app = Flask(__name__)

# Initialize Google Custom Search API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

# Initialize AI Model
try:
    # Try loading local model first
    model_name = "TheBloke/Llama-2-7B-Chat-GGML"
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )
except Exception as e:
    print(f"Warning: Could not load primary model: {e}")
    # Fallback to smaller model
    try:
        model_name = "distilgpt2"
        summarizer = pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        print(f"Error: Could not load fallback model: {e}")
        summarizer = None

def detect_file_type(url):
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    return {
        '.pdf': 'PDF',
        '.doc': 'Word',
        '.docx': 'Word',
        '.ppt': 'PowerPoint',
        '.pptx': 'PowerPoint',
        '.xls': 'Excel',
        '.xlsx': 'Excel',
        '.txt': 'Text',
    }.get(ext, 'webpage')

def extract_page_content(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text[:1000]  # First 1000 characters
    except:
        return ""

def google_search(query, num=20, file_type=None):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    
    # Add file type to query if specified
    if file_type:
        query = f"{query} filetype:{file_type}"
    
    result = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=num).execute()
    items = result.get('items', [])
    
    # Enhance results with file type and content
    enhanced_results = []
    for item in items:
        file_type = detect_file_type(item['link'])
        content = extract_page_content(item['link']) if file_type == 'webpage' else ""
        
        enhanced_results.append({
            **item,
            'file_type': file_type,
            'content': content
        })
    
    return enhanced_results

def process_with_ai(query, search_results):
    if not summarizer:
        return "AI analysis currently unavailable. Please try again later."
    
    # Group results by file type
    grouped_results = {}
    for result in search_results:
        ft = result['file_type']
        if ft not in grouped_results:
            grouped_results[ft] = []
        grouped_results[ft].append(result)
    
    # Create enhanced context
    context = f"Search Query: {query}\n\nResults Summary:\n"
    for file_type, results in grouped_results.items():
        context += f"\n{file_type} Results ({len(results)}):\n"
        for r in results[:3]:
            context += f"- {r['title']}\n  {r['snippet']}\n"
    
    try:
        # Use summarizer with appropriate parameters based on model type
        if hasattr(summarizer, "task") and summarizer.task == "summarization":
            summary = summarizer(context, max_length=150, min_length=50)[0]['summary_text']
        else:
            # For text generation models
            summary = summarizer(context, max_length=150, num_return_sequences=1)[0]['generated_text']
        
        return f"""Search Analysis:
1. Found {len(search_results)} results across {len(grouped_results)} file types
2. {summary}
3. Suggested search refinements:
   - Add filetype: for specific document types
   - Use quotes for exact phrases
   - Try more specific keywords"""
    
    except Exception as e:
        print(f"AI processing error: {e}")
        return "Could not generate AI analysis. Showing raw search results."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    file_type = request.args.get('type', None)
    
    if not query:
        return render_template('index.html')
    
    search_results = google_search(query, file_type=file_type)
    ai_response = process_with_ai(query, search_results)
    
    # Group results by file type
    grouped_results = {}
    for result in search_results:
        ft = result['file_type']
        if ft not in grouped_results:
            grouped_results[ft] = []
        grouped_results[ft].append(result)
    
    return render_template('results.html', 
                         query=query,
                         grouped_results=grouped_results,
                         ai_response=ai_response,
                         selected_type=file_type)

if __name__ == '__main__':
    app.run(debug=True)
