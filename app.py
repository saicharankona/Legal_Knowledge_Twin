import os
import io
import uuid
import logging
import time
import re
from typing import List, Optional
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Environment variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# Validate environment variables
required_vars = ['OPENROUTER_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# --- OpenRouter chat model selection -----------------------------------------
# Ordered by preference. Free-tier availability on OpenRouter rotates fairly
# often, so this is a candidate list rather than a single hardcoded model.
# We do NOT make a live test call to any of these at startup - that burns
# free-tier quota on every deploy/restart and can crash the app before it
# ever serves a request. Instead we track the current preferred candidate,
# and only fall through to the next one at REQUEST TIME if the current one
# actually fails (quota exhausted, model retired, provider error, etc).
# Check openrouter.ai/models (filter: Price -> Free) periodically and update
# this list if any of these get pulled or renamed.
OPENROUTER_MODEL_CANDIDATES = [
    'openai/gpt-oss-20b:free',
    'meta-llama/llama-4-maverick:free',
    'deepseek/deepseek-r1-distill:free',
    'openrouter/free',  # OpenRouter's own auto-router as a last-resort catch-all
]

# Index into OPENROUTER_MODEL_CANDIDATES of the model we currently believe works.
_current_model_index = 0


def generate_with_fallback(prompt: str, max_retries_per_model: int = 2) -> str:
    """
    Generate content using the current preferred OpenRouter free model,
    retrying on transient errors (rate limits) and falling back to the next
    candidate model on persistent/hard failures (quota exhausted after
    retries, model not found/retired, provider error).

    Raises the last encountered exception if every candidate is exhausted.
    """
    global _current_model_index

    last_error: Optional[Exception] = None

    for offset in range(len(OPENROUTER_MODEL_CANDIDATES)):
        model_index = (_current_model_index + offset) % len(OPENROUTER_MODEL_CANDIDATES)
        model_name = OPENROUTER_MODEL_CANDIDATES[model_index]

        for attempt in range(max_retries_per_model):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://colab.research.google.com",
                        "X-Title": "Legal Knowledge Assistant"
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    text = data['choices'][0]['message']['content']
                    if offset != 0:
                        # A fallback model worked - make it the new default so
                        # subsequent requests don't re-pay the cost of retrying
                        # the broken one first.
                        logger.info(f"Switching default OpenRouter model to: {model_name}")
                        _current_model_index = model_index
                    return text

                elif response.status_code == 429:
                    # Rate limit / quota. Worth a short retry, then move on
                    # to the next model rather than blocking the request forever.
                    last_error = Exception(f"Rate limited on {model_name}: {response.text}")
                    logger.warning(
                        f"Rate limit hit on {model_name} "
                        f"(attempt {attempt + 1}/{max_retries_per_model}): {response.text}"
                    )
                    if attempt < max_retries_per_model - 1:
                        time.sleep(20)
                        continue
                    break  # exhausted retries for this model, try next candidate

                elif response.status_code in (401, 403, 404):
                    # This model/key combo is not usable at all. No point
                    # retrying, move straight to the next candidate.
                    last_error = Exception(f"Model {model_name} unavailable ({response.status_code}): {response.text}")
                    logger.warning(f"Model {model_name} unavailable ({response.status_code}): {response.text}")
                    break

                else:
                    last_error = Exception(f"OpenRouter error on {model_name}: {response.status_code} {response.text}")
                    logger.warning(
                        f"OpenRouter API error on {model_name} "
                        f"(attempt {attempt + 1}/{max_retries_per_model}): {response.status_code} {response.text}"
                    )
                    if attempt < max_retries_per_model - 1:
                        time.sleep(5)
                        continue
                    break

            except requests.RequestException as e:
                last_error = e
                logger.warning(
                    f"Request error on {model_name} "
                    f"(attempt {attempt + 1}/{max_retries_per_model}): {e}"
                )
                if attempt < max_retries_per_model - 1:
                    time.sleep(5)
                    continue
                break

    logger.error(f"All OpenRouter model candidates failed. Last error: {last_error}")
    raise last_error if last_error else RuntimeError("OpenRouter generation failed for unknown reasons")


# Initialize Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Constants
COLLECTION_NAME = "legal_knowledge_v2"
UPLOADS_COLLECTION = "user_uploaded_v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024

# Active sessions
active_upload_sessions = {}


def clean_response(text: str) -> str:
    """Clean and format the response for professional presentation"""
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'__', '', text)
    text = re.sub(r'```', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    text = re.sub(r'^[\s]*[-•*]\s*', '  - ', text, flags=re.MULTILINE)
    return text.strip()


def get_embedding(text: str) -> List[float]:
    """Generate embeddings using OpenRouter API."""
    models_to_try = [
        "openai/text-embedding-3-small",
        "openai/text-embedding-ada-002",
        "cohere/embed-english-v3.0",
    ]

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
                    json={
                        "model": model,
                        "input": text[:8192]
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    embedding_data = response.json()
                    logger.info(f"Embedding generated using {model}")
                    return embedding_data['data'][0]['embedding']
                else:
                    if attempt < 2:
                        time.sleep(2 ** attempt)

            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)

    raise Exception("All embedding models failed")


def init_qdrant_collections():
    """Initialize Qdrant collections."""
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

    except Exception as e:
        logger.error(f"Qdrant initialization error: {e}")
        raise


# Initialize collections
init_qdrant_collections()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        qdrant_client.get_collections()
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'qdrant': 'connected',
            'collection': COLLECTION_NAME,
            'total_points': collection_info.points_count,
            'llm_model': OPENROUTER_MODEL_CANDIDATES[_current_model_index]
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500


@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process a PDF document."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400

        if not uploaded_file.filename.lower().endswith('.pdf'):
            return jsonify({'status': 'error', 'message': 'Only PDF files are supported'}), 400

        file_bytes = uploaded_file.read()
        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            return jsonify({
                'status': 'error',
                'message': f'File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB'
            }), 400

        reader = PdfReader(io.BytesIO(file_bytes))
        session_id = str(uuid.uuid4())

        raw_documents = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ''
            page_text = page_text.strip()
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

        if not raw_documents:
            return jsonify({
                'status': 'error',
                'message': 'No text could be extracted from the PDF'
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
            'chunks': len(chunked_documents)
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
    """Remove uploaded document from Qdrant."""
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id', '')

        if not session_id:
            return jsonify({'status': 'success'})

        # Scroll through all points in uploads collection
        scroll_result = qdrant_client.scroll(
            collection_name=UPLOADS_COLLECTION,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        point_ids = []
        for point in scroll_result[0]:
            if point.payload.get('session_id') == session_id:
                point_ids.append(point.id)

        # Delete points by ID
        if point_ids:
            for point_id in point_ids:
                try:
                    qdrant_client.delete(
                        collection_name=UPLOADS_COLLECTION,
                        points_selector=qdrant_models.PointIdsList(points=[point_id])
                    )
                except Exception as e:
                    logger.warning(f"Could not delete point {point_id}: {e}")

        active_upload_sessions.pop(session_id, None)
        logger.info(f"Removed document for session {session_id}")

        return jsonify({'status': 'success'})

    except Exception as e:
        logger.error(f"Remove document error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries with RAG."""
    try:
        data = request.get_json(silent=True) or {}
        question = data.get('question', '').strip()
        session_id = data.get('session_id', '')

        if not question:
            return jsonify({'status': 'error', 'message': 'Question is required'}), 400

        using_uploaded = session_id in active_upload_sessions

        # Generate query embedding
        query_embedding = get_embedding(question)

        # Always search the main collection for now
        # For uploaded documents, we'll filter later
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=10,
            with_payload=True
        )

        # If using uploaded documents, also search in uploads collection
        uploaded_results = []
        if using_uploaded:
            try:
                uploaded_results = qdrant_client.search(
                    collection_name=UPLOADS_COLLECTION,
                    query_vector=query_embedding,
                    limit=10,
                    with_payload=True
                )
                # Filter by session_id
                uploaded_results = [r for r in uploaded_results if r.payload.get('session_id') == session_id]
            except Exception as e:
                logger.warning(f"Uploaded search error: {e}")

        # Combine results - prioritize uploaded if using_uploaded
        if using_uploaded and uploaded_results:
            results = uploaded_results[:5]
            source_mode = 'uploaded_document'
        else:
            results = search_results[:5]
            source_mode = 'knowledge_base'

        if not results:
            return jsonify({
                'status': 'success',
                'answer': 'I could not find relevant information in the documents to answer your question.',
                'sources': [],
                'source_mode': source_mode
            }), 200

        # Prepare context
        context_parts = []
        sources = []
        for hit in results:
            payload = hit.payload
            text = payload.get('text', '')
            source = payload.get('source', 'Unknown')
            page = payload.get('page', 0)

            context_parts.append(f"[Source: {source} | Page {page + 1}]\n{text}")
            if source not in sources:
                sources.append(source)

        context = '\n\n---\n\n'.join(context_parts)

        # Generate response with OpenRouter (free model)
        prompt = f"""You are Legal Knowledge Twin, an expert AI assistant for Indian law.

You MUST answer ONLY from the supplied legal context.

Grounding Rules:
- Every factual statement must be directly supported by the context.
- Do not use prior knowledge.
- Do not assume missing facts.
- Do not invent section numbers or case names.
- If information is missing, explicitly state that it is not available.
- Never speculate.

If the answer cannot be supported by the provided context, reply ONLY:
"Sorry, I could not find this information in the uploaded legal documents."

=====================
LEGAL CONTEXT
=====================

{context}

=====================
QUESTION
=====================

{question}

=====================
OUTPUT FORMAT
=====================

SUMMARY:
(2-3 sentence answer)

LEGAL ANALYSIS:
(Explain only what is supported by the context.)

APPLICABLE LEGAL PROVISIONS:
- Section / Act
- Explanation

PRACTICAL GUIDANCE:
- Rights
- Obligations
- Procedure
- Limitations

SOURCES:
List every document cited.

DISCLAIMER:
This answer is generated solely from the uploaded legal documents and is not a substitute for professional legal advice.

Begin your response.
"""

        try:
            answer = generate_with_fallback(prompt)
        except Exception as e:
            logger.error(f"All OpenRouter models failed for /chat: {e}")
            return jsonify({
                'status': 'error',
                'message': 'The AI service is temporarily unavailable (rate limit or access issue). Please try again shortly.'
            }), 503

        return jsonify({
            'status': 'success',
            'answer': answer,
            'sources': sources,
            'source_mode': source_mode
        }), 200

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
