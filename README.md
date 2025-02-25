# Nova Lite

An AI-powered search engine with intelligent summarization and region-aware results.

## Features

- AI-powered search summaries
- Smart file type detection
- Regional search awareness
- Dark/light theme support
- Fast and efficient search
- Local model caching
- Markdown-formatted results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kanopusdev/nova-lite.git
cd nova-lite
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Google Custom Search:
- Go to [Google Cloud Console](https://console.cloud.google.com)
- Create a new project
- Enable Custom Search API
- Create API credentials
- Copy your API key

5. Create Search Engine:
- Visit [Programmable Search Engine](https://programmablesearchengine.google.com)
- Create new search engine
- Get your Search Engine ID

6. Configure settings:
- Open the app in browser
- Click settings icon
- Enter your API key and Search Engine ID
- Save settings

## Usage

1. Start the server:
```bash
python app.py
```

2. Open http://localhost:5000 in your browser

3. Search Features:
- Simple search: Enter query and press search
- File type filter: Use dropdown to filter results
- Sections: Choose between All/Images/Shopping/News
- AI Summary: Get AI-powered analysis for complex queries

4. Advanced Search:
- Use quotes for exact matches: `"exact phrase"`
- Exclude terms: `-excludedterm`
- Site specific: `site:example.com`
- File types: Use the dropdown or `filetype:pdf`

## Technical Details

### Components:
- **Frontend**: Flask + Jinja2 + TailwindCSS
- **Backend**: Python Flask
- **AI**: Transformers (BART-large-CNN)
- **Search**: Google Custom Search API
- **Caching**: Local file system cache

### Performance:
- Model caching for faster startup
- Search result caching (24h)
- Optimized AI processing
- Regional URL resolution
- Efficient memory usage

### Security:
- Secure API key storage
- Session management
- Input sanitization
- Error handling

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## License

This project is licensed under the Nova Lite Search Engine License (NLS v1.0).
See LICENSE.md for details.

## Acknowledgments

- Transformers by Hugging Face
- Google Custom Search API
- Flask Framework
- TailwindCSS

---

Made with ❤️ by Nova Lite Contributors