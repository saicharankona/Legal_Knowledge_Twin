import os
import io
import uuid
import logging
from typing import List
from datetime import datetime
from pathlib import Path
 
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time
import re
from PIL import Image
import pypdfium2 as pdfium
 
load_dotenv()
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
 
# Environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
 
# Validate environment variables
required_vars = ['GEMINI_API_KEY', 'OPENROUTER_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
 
# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
 
# Initialize Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
 
# Constants
COLLECTION_NAME = "legal_knowledge_v2"
UPLOADS_COLLECTION = "user_uploaded_v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024
 
ALLOWED_PDF_EXTENSIONS = ('.pdf',)
ALLOWED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
ALLOWED_UPLOAD_EXTENSIONS = ALLOWED_PDF_EXTENSIONS + ALLOWED_IMAGE_EXTENSIONS
 
# Below this many characters of extracted text, a PDF page is treated as a scan/
# photo rather than a real text page, and gets OCR'd via Gemini Vision instead.
MIN_TEXT_LENGTH_FOR_NATIVE_EXTRACTION = 20
 
MIN_RELEVANCE_SCORE = float(os.getenv('MIN_RELEVANCE_SCORE', '0.0'))
 
NOT_FOUND_MESSAGE = "Sorry! The provided legal documents do not contain information about this specific question."
 
active_upload_sessions = {}
 
 
def clean_response(text: str) -> str:
    """Clean and format the response for professional presentation"""
    # Remove markdown formatting
    text = text.replace('**', '').replace('__', '').replace('```', '').replace('`', '')
 
    # Remove headers (##, ###, etc.)
    text = re.sub(r'#{1,6}\s*', '', text)
 
    # Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
 
    # Add space after periods
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
 
    # Fix bullet points
    text = re.sub(r'^[\s]*[-•*]\s*', '  - ', text, flags=re.MULTILINE)
 
    return text.strip()
 
 
def format_official_response(answer: str, sources: list) -> str:
    """Format the response in official legal document style"""
    cleaned_answer = clean_response(answer)
    response = []
    response.append("LEGAL KNOWLEDGE TWIN")
    response.append("")
    response.append(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    response.append("")
    response.append("DISCLAIMER: This information is for educational purposes only")
    response.append("and does not constitute legal advice. Consult a qualified")
    response.append("attorney for legal matters.")
    response.append("")
    response.append("")
 
    sections = cleaned_answer.split('\n\n')
    for section in sections:
        if section.strip():
            if any(keyword in section.lower() for keyword in ['summary', 'explanation', 'sources', 'guidance', 'action']):
                lines = section.split('\n')
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in ['summary:', 'explanation:', 'sources:', 'guidance:', 'action:']):
                        header = line.upper()
                        lines[i] = f"\n{header}\n{'-' * len(header)}"
                section = '\n'.join(lines)
            response.append(section)
            response.append("")
 
    if sources and not any('source' in s.lower() for s in sections):
        response.append("SOURCES REFERENCED")
        response.append("-" * 20)
        for i, source in enumerate(sources, 1):
            response.append(f"  {i}. {source}")
        response.append("")
 
    response.append("Legal Knowledge Twin | AI-Powered Legal Assistant")
    response.append("Based on Indian Legal Documents")
 
    return '\n'.join(response)
 
 
def ocr_extract_text_with_gemini(image: Image.Image) -> str:
    """
    Extract text from an image using Gemini's vision capability.
    Used for: uploaded photos (jpg/png/etc.) and scanned PDF pages that have
    no embedded text layer. No system OCR binary (e.g. Tesseract) is required,
    which matters because Render's native Python environment can't install one.
    """
    try:
        response = gemini_model.generate_content([
            "Extract all readable text from this image exactly as written. "
            "Preserve paragraph breaks where meaningful. Output ONLY the "
            "extracted text — no commentary, no labels, no markdown.",
            image
        ])
        return (response.text or "").strip()
    except Exception as e:
        logger.warning(f"Gemini OCR failed: {e}")
        return ""
 
 
def get_embedding(text: str) -> List[float]:
    """Generate embeddings using OpenRouter API."""
    models_to_try = [
        "openai/text-embedding-3-small",
        "openai/text-embedding-ada-002",
        "cohere/embed-english-v3.0",
    ]
    last_error = None
    for model in models_to_try:
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://colab.research.google.com",
                        "X-Title": "Legal Knowledge Assistant"
                    },
                    json={"model": model, "input": text[:8192]},
                    timeout=60
                )
                if response.status_code == 200:
                    embedding_data = response.json()
                    logger.info(f"Embedding generated using {model}")
                    return embedding_data['data'][0]['embedding']
                else:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    last_error = f"Status {response.status_code}: {response.text[:100]}"
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                last_error = str(e)
    raise Exception(f"All embedding models failed: {last_error}")
 
 
def ask_llm(prompt: str) -> str:
    """Try each model until one succeeds."""
    chat_models = [
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-7b-instruct",
        "nvidia/llama-3.2-nemotron-1b-v2",
        "openai/gpt-4o-mini",
    ]
    system_prompt = f"""You are a document-grounded legal assistant for Indian law.
 
RULES:
1. Answer using the information contained in the LEGAL CONTEXT provided in the user message.
   The context may use different wording than the question — read it carefully and use any
   information that is relevant, even if phrased differently or spread across the context.
2. Do NOT add legal knowledge from your own training that is not present in the context —
   don't introduce sections, acts, or facts that aren't stated in the supplied context.
3. Only respond with the exact fallback sentence below if the context truly does not address
   the topic of the question at all. If the context is on-topic but only partially answers
   the question, answer with what the context does cover — do not refuse just because the
   context isn't a perfect or complete match.
4. Fallback sentence (use word-for-word, only when the context is genuinely unrelated to the question):
   "{NOT_FOUND_MESSAGE}"
 
When the context is relevant, provide responses in official, formal language.
Do not use markdown, emojis, or special symbols.
Structure your response with these sections:
SUMMARY: Brief overview of the legal position, based on the context
DETAILED EXPLANATION: Comprehensive explanation using the supplied context
LEGAL PROVISIONS: Sections/provisions that appear in the context
SOURCES: List of documents referenced (from the context)
PRACTICAL GUIDANCE: Actionable next steps for the user, based on the context"""
 
    for model in chat_models:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://colab.research.google.com",
                    "X-Title": "Legal Knowledge Twin"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1200,
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                logger.info(f"Answered by: {model}")
                return answer
            else:
                logger.warning(f"Model {model} failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Model {model} failed: {str(e)[:60]}")
            continue
    return NOT_FOUND_MESSAGE
 
 
def init_qdrant_collections():
    """Initialize Qdrant collections and required payload indexes."""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
 
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creating {COLLECTION_NAME} collection...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=1536,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            logger.info(f"Created {COLLECTION_NAME} collection")
        else:
            logger.info(f"Collection {COLLECTION_NAME} already exists")
 
        if UPLOADS_COLLECTION not in collection_names:
            logger.info(f"Creating {UPLOADS_COLLECTION} collection...")
            qdrant_client.create_collection(
                collection_name=UPLOADS_COLLECTION,
                vectors_config=qdrant_models.VectorParams(
                    size=1536,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            logger.info(f"Created {UPLOADS_COLLECTION} collection")
        else:
            logger.info(f"Collection {UPLOADS_COLLECTION} already exists")
 
        # IMPORTANT: Qdrant requires a payload index on any field used inside a
        # query_filter (e.g. session_id) before you can filter/search on it.
        # Without this, /chat with an uploaded document raises:
        #   400 Bad Request: Index required but not found for "session_id" ...
        # create_payload_index is idempotent-safe to call on every startup —
        # if the index already exists, Qdrant just leaves it as-is.
        try:
            qdrant_client.create_payload_index(
                collection_name=UPLOADS_COLLECTION,
                field_name='session_id',
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Ensured payload index on 'session_id' for {UPLOADS_COLLECTION}")
        except Exception as e:
            logger.warning(f"Payload index creation for 'session_id' skipped/failed: {e}")
 
    except Exception as e:
        logger.error(f"Qdrant initialization error: {e}")
        raise
 
 
init_qdrant_collections()
 
 
@app.route('/')
def home():
    return render_template('index.html')
 
 
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)
 
 
@app.route('/health', methods=['GET'])
def health_check():
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'qdrant': 'connected',
            'collection': COLLECTION_NAME,
            'total_points': collection_info.points_count
        }), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
 
 
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        question = data.get('question', '').strip()
        session_id = data.get('session_id', '')
 
        if not question:
            return jsonify({'status': 'error', 'message': 'Question is required'}), 400
 
        using_uploaded = session_id in active_upload_sessions
        collection_name = UPLOADS_COLLECTION if using_uploaded else COLLECTION_NAME
 
        filter_condition = None
        if using_uploaded:
            filter_condition = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key='session_id',
                        match=qdrant_models.MatchValue(value=session_id)
                    )
                ]
            )
 
        query_embedding = get_embedding(question)
 
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=6,
            query_filter=filter_condition,
            with_payload=True
        )
 
        if search_results:
            top_scores = [round(hit.score, 4) for hit in search_results]
            logger.info(f"Retrieved scores for question '{question[:50]}...': {top_scores}")
        else:
            logger.info(f"No Qdrant results at all for question '{question[:50]}...'")
 
        search_results = [hit for hit in search_results if hit.score >= MIN_RELEVANCE_SCORE]
 
        if not search_results:
            return jsonify({
                'status': 'success',
                'answer': NOT_FOUND_MESSAGE,
                'sources': [],
                'source_mode': 'uploaded_document' if using_uploaded else 'knowledge_base'
            }), 200
 
        context_parts = []
        sources = []
        for hit in search_results:
            payload = hit.payload
            text = payload.get('text', '')
            source = payload.get('source', 'Unknown')
            page = payload.get('page', 0)
            context_parts.append(f"[Source: {source} | Page {page + 1}]\n{text}")
            if source not in sources:
                sources.append(source)
 
        context = '\n\n---\n\n'.join(context_parts)
 
        prompt = f"""LEGAL CONTEXT (use this to answer the question below):
{context}
 
USER QUESTION:
{question}
 
Answer using the LEGAL CONTEXT above. The context may not use the exact same words as the
question — read carefully and use anything relevant, even if phrased differently. Do not add
outside legal knowledge that isn't stated above. Only respond with the exact fallback sentence
below if the context is genuinely unrelated to what's being asked:
"{NOT_FOUND_MESSAGE}"
 
Otherwise, use this format (formal language, no symbols):
SUMMARY:
Begin with a concise summary of the legal position in 2 to 3 sentences, drawn only from the context.
 
DETAILED EXPLANATION:
Provide a comprehensive explanation using only what appears in the context, including:
- The legal framework and its purpose, as stated in the context
- Key provisions and their interpretation, as stated in the context
- Applicable conditions and exceptions, as stated in the context
 
LEGAL PROVISIONS:
List only the specific sections, acts, or provisions that literally appear in the context above.
 
PRACTICAL GUIDANCE:
Explain what this means for the user, based only on the context, including:
- Rights and obligations under the law as described in the context
- Recommended actions
- Important considerations
 
RESPONSE:"""
 
        answer = ask_llm(prompt)
 
        context_words = set(re.findall(r'[a-z]{5,}', context.lower()))
        answer_words = set(re.findall(r'[a-z]{5,}', answer.lower()))
        overlap = len(context_words & answer_words)
        overlap_ratio = overlap / len(context_words) if context_words else 1
        logger.info(f"Context/answer overlap: {overlap} words ({overlap_ratio:.1%} of context vocab)")
 
        formatted_answer = format_official_response(answer, sources)
 
        return jsonify({
            'status': 'success',
            'answer': formatted_answer,
            'sources': sources,
            'source_mode': 'uploaded_document' if using_uploaded else 'knowledge_base'
        }), 200
 
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
 
 
@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
 
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
 
        filename_lower = uploaded_file.filename.lower()
        if not filename_lower.endswith(ALLOWED_UPLOAD_EXTENSIONS):
            return jsonify({
                'status': 'error',
                'message': 'Only PDF and image files (jpg, jpeg, png, webp, bmp) are supported'
            }), 400
 
        file_bytes = uploaded_file.read()
        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            return jsonify({
                'status': 'error',
                'message': f'File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB'
            }), 400
 
        session_id = str(uuid.uuid4())
        raw_documents = []
        is_pdf = filename_lower.endswith(ALLOWED_PDF_EXTENSIONS)
 
        if is_pdf:
            # Try normal text extraction first (fast, free, works for native/digital PDFs)
            reader = PdfReader(io.BytesIO(file_bytes))
            pdf_for_render = None  # only opened with pdfium if OCR fallback is needed
 
            for page_num, page in enumerate(reader.pages):
                page_text = (page.extract_text() or '').strip()
 
                if len(page_text) < MIN_TEXT_LENGTH_FOR_NATIVE_EXTRACTION:
                    # Likely a scanned/photographed page with no real text layer —
                    # render it to an image and OCR it with Gemini Vision instead.
                    if pdf_for_render is None:
                        pdf_for_render = pdfium.PdfDocument(file_bytes)
                    try:
                        bitmap = pdf_for_render[page_num].render(scale=2.0)
                        page_image = bitmap.to_pil()
                        page_text = ocr_extract_text_with_gemini(page_image).strip()
                        if page_text:
                            logger.info(f"OCR'd page {page_num + 1} of {uploaded_file.filename}")
                    except Exception as e:
                        logger.warning(f"OCR render failed on page {page_num + 1}: {e}")
 
                if page_text:
                    raw_documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                'source': uploaded_file.filename,
                                'page': page_num,
                                'session_id': session_id
                            }
                        )
                    )
        else:
            # Plain photo/image upload — OCR the whole image via Gemini Vision.
            try:
                image = Image.open(io.BytesIO(file_bytes))
                image.load()
            except Exception:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not read the uploaded image. Please upload a valid jpg, png, webp, or bmp file.'
                }), 400
 
            page_text = ocr_extract_text_with_gemini(image).strip()
            if page_text:
                raw_documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            'source': uploaded_file.filename,
                            'page': 0,
                            'session_id': session_id
                        }
                    )
                )
 
        if not raw_documents:
            return jsonify({
                'status': 'error',
                'message': 'No text could be extracted from the file (including via OCR)'
            }), 400
 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=['\n\n', '\n', '.', ' ', '']
        )
        chunked_documents = splitter.split_documents(raw_documents)
 
        points = []
        collection_info = qdrant_client.get_collection(UPLOADS_COLLECTION)
        start_id = collection_info.points_count
 
        for i, doc in enumerate(chunked_documents):
            embedding = get_embedding(doc.page_content[:8000])
            point_id = start_id + i
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'text': doc.page_content,
                        'source': doc.metadata.get('source', 'Unknown'),
                        'page': doc.metadata.get('page', 0),
                        'session_id': session_id,
                        'chunk_index': i
                    }
                )
            )
 
        qdrant_client.upsert(collection_name=UPLOADS_COLLECTION, points=points)
        active_upload_sessions[session_id] = {
            'filename': uploaded_file.filename,
            'chunks': len(chunked_documents),
            'timestamp': datetime.now()
        }
 
        logger.info(f"Uploaded {uploaded_file.filename} with {len(chunked_documents)} chunks")
 
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'filename': uploaded_file.filename,
            'chunks_indexed': len(chunked_documents)
        }), 200
 
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
 
 
@app.route('/remove-document', methods=['POST'])
def remove_document():
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id', '')
 
        if not session_id:
            return jsonify({'status': 'success'})
 
        qdrant_client.delete(
            collection_name=UPLOADS_COLLECTION,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key='session_id',
                            match=qdrant_models.MatchValue(value=session_id)
                        )
                    ]
                )
            )
        )
        active_upload_sessions.pop(session_id, None)
        logger.info(f"Removed document for session {session_id}")
        return jsonify({'status': 'success'})
 
    except Exception as e:
        logger.error(f"Remove document error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
 
 
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe a short voice recording (webm/opus from the browser's MediaRecorder)
    into text using Gemini's audio understanding. No separate speech-to-text
    service or system binary (e.g. ffmpeg/whisper.cpp) is required.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio provided'}), 400
 
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
 
        if not audio_bytes:
            return jsonify({'status': 'error', 'message': 'Empty audio recording'}), 400
 
        response = gemini_model.generate_content([
            "Transcribe the speech in this audio recording exactly as spoken. "
            "Output ONLY the transcript text — no commentary, no timestamps, "
            "no speaker labels, no quotation marks.",
            {"mime_type": "audio/webm", "data": audio_bytes}
        ])
        transcript = (response.text or "").strip()
 
        if not transcript:
            return jsonify({
                'status': 'error',
                'message': 'Could not transcribe audio. Please try again and speak clearly.'
            }), 400
 
        logger.info(f"Transcribed voice input: {transcript[:80]}")
        return jsonify({'status': 'success', 'transcript': transcript}), 200
 
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
 
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
