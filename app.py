from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import re
from dotenv import load_dotenv
from urllib.parse import urlparse, urljoin
import mimetypes
import requests
from bs4 import BeautifulSoup
import torch
import socket
import geoip2.database
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

app = Flask(__name__)

# Initialize Google Custom Search API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

# Initialize GeoIP reader
try:
    GEOIP_READER = geoip2.database.Reader('GeoLite2-Country.mmdb')
except:
    GEOIP_READER = None

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

def get_user_region(request):
    try:
        if GEOIP_READER:
            # Get client IP
            if request.headers.get('X-Forwarded-For'):
                ip = request.headers['X-Forwarded-For'].split(',')[0]
            else:
                ip = request.remote_addr
                
            # Get country from IP
            response = GEOIP_READER.country(ip)
            return response.country.iso_code.lower()
    except:
        pass
    return None

def resolve_regional_url(url, region=None):
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return url
            
        # Common regional patterns
        regional_domains = {
            'google.com': {'in': 'google.co.in', 'uk': 'google.co.uk'},
            'amazon.com': {'in': 'amazon.in', 'uk': 'amazon.co.uk'},
            # Add more as needed
        }
        
        domain = parsed.netloc.lower()
        base_domain = '.'.join(domain.split('.')[-2:])
        
        if base_domain in regional_domains and region:
            regional_domain = regional_domains[base_domain].get(region)
            if regional_domain:
                return urljoin(f"https://{regional_domain}", parsed.path)
    except:
        pass
    return url

def google_search(query, num=20, file_type=None, region=None):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        
        # Validate credentials
        if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
            raise ValueError("Missing Google API credentials")
        
        # Clean and prepare query
        query = query.strip()
        if not query:
            return []
        
        search_query = query
        if file_type:
            search_query = f"{query} filetype:{file_type}"
        
        try:
            result = service.cse().list(
                q=search_query,
                cx=SEARCH_ENGINE_ID,
                num=min(num, 10),  # Google CSE free tier limit
                safe='active',  # Safe search
                gl=region.upper() if region else None  # Add country restriction
            ).execute()
            
            items = result.get('items', [])
            
        except Exception as e:
            print(f"Search API error: {e}")
            # Return empty results on API error
            return []
        
        # Enhance results with file type and content
        enhanced_results = []
        for item in items:
            try:
                # Resolve regional URL
                item['link'] = resolve_regional_url(item['link'], region)
                file_type = detect_file_type(item['link'])
                content = extract_page_content(item['link']) if file_type == 'webpage' else ""
                
                enhanced_results.append({
                    **item,
                    'file_type': file_type,
                    'content': content
                })
            except Exception as e:
                print(f"Error processing result: {e}")
                continue
        
        return enhanced_results
        
    except Exception as e:
        print(f"Google Search error: {e}")
        return []

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
        input_length = len(context.split())
        max_length = min(input_length - 10, 100)  # Ensure shorter summary
        min_length = max(30, max_length // 2)  # Dynamic min length
        
        if hasattr(summarizer, "task") and summarizer.task == "summarization":
            summary = summarizer(context, 
                               max_length=max_length,
                               min_length=min_length,
                               do_sample=False)[0]['summary_text']
        else:
            summary = summarizer(context, 
                               max_length=max_length,
                               num_return_sequences=1,
                               do_sample=False)[0]['generated_text']
        
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
    query = request.args.get('q', '').strip()
    file_type = request.args.get('type', None)
    
    if not query:
        return render_template('index.html')
    
    try:
        # Get user's region
        region = get_user_region(request)
        search_results = google_search(query, file_type=file_type, region=region)
        
        if not search_results:
            error_message = "No results found. Please try different keywords."
            if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
                error_message = "Search configuration error. Please check API settings."
            return render_template('results.html',
                                query=query,
                                grouped_results={},
                                ai_response=error_message,
                                selected_type=file_type,
                                error=True)
        
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
                             
    except Exception as e:
        print(f"Search route error: {e}")
        return render_template('results.html',
                             query=query,
                             grouped_results={},
                             ai_response="An error occurred during search. Please try again.",
                             selected_type=file_type,
                             error=True)

if __name__ == '__main__':
    app.run(debug=True)
