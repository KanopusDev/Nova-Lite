from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from googleapiclient.discovery import build
from transformers import pipeline, AutoTokenizer  # Simplified imports
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
import tldextract
import dns.resolver
import concurrent.futures
import reverse_geocoder as rg
import functools
import hashlib
import pickle
from pathlib import Path
import json
import markdown2
from textwrap import dedent
warnings.filterwarnings('ignore')

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load settings from file
def load_settings():
    try:
        settings_file = Path(__file__).parent / 'settings.json'
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    return {'api_key': '', 'search_engine_id': ''}

def save_settings_to_file(settings):
    try:
        settings_file = Path(__file__).parent / 'settings.json'
        with open(settings_file, 'w') as f:
            json.dump(settings, f)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# Update API key initialization
settings = load_settings()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or settings.get('api_key')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID') or settings.get('search_engine_id')

# Initialize Google Custom Search API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

# Initialize GeoIP reader
try:
    GEOIP_READER = geoip2.database.Reader('GeoLite2-Country.mmdb')
except:
    GEOIP_READER = None

# Constants
CACHE_DIR = Path(__file__).parent / "cache"
MODEL_CACHE_DIR = CACHE_DIR / "models"
SEARCH_CACHE_DIR = CACHE_DIR / "search"

# Create cache directories
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Performance optimizations
torch.set_grad_enabled(False)  # Disable gradients
torch.set_num_threads(4)  # Limit CPU threads
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Global model state
MODEL_STATE = {
    'initialized': False,
    'summarizer': None,
    'error': None
}

def initialize_models():
    """Pre-initialize all models before first request"""
    global MODEL_STATE
    try:
        print("Initializing AI models...")
        model_path = MODEL_CACHE_DIR / "bart-large-cnn"
        
        if not model_path.exists():
            print("Downloading models (this may take a few minutes)...")
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1,
                cache_dir=model_path,
                model_kwargs={"low_cpu_mem_usage": True}
            )
        else:
            print("Loading cached models...")
            summarizer = pipeline(
                "summarization",
                model=str(model_path),
                device=-1,
                model_kwargs={"low_cpu_mem_usage": True}
            )
        
        MODEL_STATE['summarizer'] = summarizer
        MODEL_STATE['initialized'] = True
        print("Models initialized successfully!")
        
    except Exception as e:
        MODEL_STATE['error'] = str(e)
        print(f"Model initialization error: {e}")

# Initialize models before first request
@app.before_first_request
def setup():
    initialize_models()

def get_model_status():
    """Get current model initialization status"""
    return {
        'initialized': MODEL_STATE['initialized'],
        'error': MODEL_STATE['error']
    }

# Add search result caching
def cache_key(query, **kwargs):
    cache_str = f"{query}:{kwargs}"
    return hashlib.md5(cache_str.encode()).hexdigest()

def cache_results(func):
    @functools.wraps(func)
    def wrapper(query, *args, **kwargs):
        key = cache_key(query, **kwargs)
        cache_file = SEARCH_CACHE_DIR / f"{key}.pickle"
        
        # Check cache expiry (24 hours)
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        results = func(query, *args, **kwargs)
        
        # Cache results
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    return wrapper

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

def get_regional_domains(domain):
    try:
        # Extract base domain
        ext = tldextract.extract(domain)
        base_domain = f"{ext.domain}.{ext.suffix}"
        
        # Common regional TLD patterns
        regional_tlds = [
            'co.uk', 'co.jp', 'co.in', 'com.au', 'co.nz',
            'de', 'fr', 'es', 'it', 'ca', 'br', 'mx',
            'ru', 'cn', 'jp', 'kr', 'in', 'au', 'nz'
        ]
        
        # Check domain variants in parallel
        domains = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            # Check main domain
            futures.append(executor.submit(check_domain, f"{ext.domain}.com"))
            # Check regional variants
            for tld in regional_tlds:
                test_domain = f"{ext.domain}.{tld}"
                futures.append(executor.submit(check_domain, test_domain))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    domains.append(future.result())
        
        return domains
    except Exception as e:
        print(f"Error getting regional domains: {e}")
        return []

def check_domain(domain):
    try:
        answers = dns.resolver.resolve(domain, 'A')
        if answers:
            return domain
    except:
        return None

def resolve_regional_url(url, region=None):
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return url
            
        # Extract domain parts
        ext = tldextract.extract(parsed.netloc)
        base_domain = f"{ext.domain}.{ext.suffix}"
        
        # Get all regional variants
        regional_domains = get_regional_domains(base_domain)
        
        # If we have a specific region, try to match it
        if region and regional_domains:
            for domain in regional_domains:
                if region.lower() in domain:
                    return urljoin(f"https://{domain}", parsed.path)
        
        # Otherwise return original URL
        return url
        
    except Exception as e:
        print(f"URL resolution error: {e}")
        return url

@cache_results
def google_search(query, num=20, file_type=None, section=None, region=None):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        
        if not query.strip():
            return []
            
        # Add file type if specified
        search_query = f"{query.strip()} filetype:{file_type}" if file_type else query.strip()
        
        # Build search parameters
        search_params = {
            'q': search_query,
            'cx': SEARCH_ENGINE_ID,
            'num': min(num, 10),
            'safe': 'off'
        }
        
        # Add location if available
        if region:
            search_params.update({
                'gl': region.upper(),  # Geographic location
                'cr': f"country{region.upper()}"  # Country restrict
            })
        
        # Add section-specific parameters
        if section == 'images':
            search_params['searchType'] = 'image'
        elif section == 'news':
            search_params['dateRestrict'] = 'd7'
        
        # Execute search
        result = service.cse().list(**search_params).execute()
        
        # Process results based on section
        if section == 'images':
            return [{
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'thumbnail': item.get('image', {}).get('thumbnailLink', ''),
                'context': item.get('image', {}).get('contextLink', '')
            } for item in result.get('items', [])]
            
        elif section == 'shopping':
            return [{
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'price': item.get('pagemap', {}).get('offer', [{}])[0].get('price', 'N/A'),
                'merchant': item.get('pagemap', {}).get('organization', [{}])[0].get('name', ''),
                'image': item.get('pagemap', {}).get('cse_image', [{}])[0].get('src', '')
            } for item in result.get('items', [])]
            
        else:
            items = result.get('items', [])
            enhanced_results = []
            
            for item in items:
                try:
                    # Resolve regional URL with the passed region parameter
                    regional_url = resolve_regional_url(item.get('link', ''), region)
                    
                    enhanced_results.append({
                        'title': item.get('title', ''),
                        'link': regional_url,
                        'original_link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'file_type': detect_file_type(regional_url),
                        'mime': item.get('mime', ''),
                        'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('date', ''),
                        'content': extract_page_content(regional_url) if detect_file_type(regional_url) == 'webpage' else ''
                    })
                except Exception as e:
                    print(f"Result processing error: {e}")
                    continue
                    
            return enhanced_results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def process_with_ai(query, search_results):
    if not search_results:
        return None
    
    # Check model status
    if not MODEL_STATE['initialized']:
        return "AI models are still initializing. Please try again in a moment."
    
    if MODEL_STATE['error']:
        return f"AI analysis unavailable: {MODEL_STATE['error']}"
        
    try:
        snippets = [r['snippet'] for r in search_results[:3] if r.get('snippet')]
        if not snippets:
            return None
            
        context = f"Search: {query}\n\n" + "\n".join(snippets)
        
        try:
            chunks = [context[i:i+512] for i in range(0, len(context), 512)]
            summaries = []
            
            for chunk in chunks:
                result = MODEL_STATE['summarizer'](
                    chunk,
                    max_length=130,
                    min_length=30,
                    do_sample=False,
                    num_beams=1
                )
                if result:
                    summaries.append(result[0]['summary_text'])
            
            if not summaries:
                return None
                
            summary = " ".join(summaries)
            
            # Format summary as markdown
            markdown_summary = dedent(f"""
                ## Search Summary

                {summary}

                ### Key Information
                
                * **Query**: {query}
                * **Results**: {len(search_results)} found
                * **Types**: {', '.join(set(r['file_type'] for r in search_results))}

                ### Search Tips
                
                * Use `"quotes"` for exact matches
                * Try `-exclude` to remove terms
                * Use `site:example.com` to search specific sites
                * Filter by file type using the dropdown
            """)
            
            # Convert markdown to HTML
            return markdown2.markdown(
                markdown_summary,
                extras=['fenced-code-blocks', 'tables', 'break-on-newline']
            )
                
        except Exception as e:
            print(f"Summarization error: {e}")
            return None
        
    except Exception as e:
        print(f"AI processing error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_location', methods=['POST'])
def update_location():
    try:
        data = request.json
        lat, lon = data['lat'], data['lon']
        
        # Get location details
        location = rg.search((lat, lon))[0]
        
        # Store in session
        session['user_location'] = {
            'lat': lat,
            'lon': lon,
            'country': location['cc'],
            'city': location['name']
        }
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    file_type = request.args.get('type', None)
    section = request.args.get('section', None)
    
    if not query:
        return render_template('index.html')
    
    try:
        # Get user's region before search
        region = None
        if 'user_location' in session:
            region = session['user_location'].get('country')
        if not region:
            region = get_user_region(request)
        
        # Pass region to search function
        results = google_search(query, file_type=file_type, section=section, region=region)
        
        if not results:
            return render_template('results.html',
                                query=query,
                                error="No results found. Try different keywords.",
                                section=section,
                                grouped_results={})
        
        if section == 'images':
            return render_template('results.html',
                                query=query,
                                section=section,
                                image_results=results)
                                
        elif section == 'shopping':
            return render_template('results.html',
                                query=query,
                                section=section,
                                shopping_results=results)
        
        else:
            # Process with AI only for informational queries
            ai_response = None
            if len(query.split()) > 2:  # Only process complex queries
                ai_response = process_with_ai(query, results)
            
            # Group results
            grouped_results = {}
            for result in results:
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
        return render_template('results.html',
                             query=query,
                             error=f"Search error: {str(e)}",
                             section=section,
                             grouped_results={})

@app.route('/settings')
def settings_page():
    settings = load_settings()
    return render_template('settings.html',
                         api_key=settings.get('api_key', ''),
                         search_engine_id=settings.get('search_engine_id', ''))

@app.route('/settings/save', methods=['POST'])
def save_settings():
    global GOOGLE_API_KEY, SEARCH_ENGINE_ID
    
    api_key = request.form.get('api_key')
    search_engine_id = request.form.get('search_engine_id')
    
    if api_key and search_engine_id:
        settings = {
            'api_key': api_key,
            'search_engine_id': search_engine_id
        }
        if save_settings_to_file(settings):
            GOOGLE_API_KEY = api_key
            SEARCH_ENGINE_ID = search_engine_id
            return redirect(url_for('home'))
    
    return redirect(url_for('settings_page'))

@app.route('/api/model-status')
def model_status():
    """API endpoint to check model status"""
    return jsonify(get_model_status())

if __name__ == '__main__':
    app.run(debug=True)
