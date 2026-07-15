

import os
import io
import sys
import uuid
import tempfile

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from pypdf import PdfReader

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize


# -------------------------------------------------------
# Flask App
# -------------------------------------------------------

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# ALLOWED_ORIGINS lets you lock CORS down to your real frontend domain
# in production by setting it as a Render env var, e.g.
# ALLOWED_ORIGINS=https://legal-twin.onrender.com
# If it's not set, CORS stays open ("*") so nothing breaks by default.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
CORS(app, origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != "*" else "*")


# -------------------------------------------------------
# Environment Variables
# -------------------------------------------------------
# These must be set as Environment Variables in the Render dashboard
# (Settings -> Environment), not in userdata/Colab secrets, and not
# hardcoded here. Render injects them into os.environ at container
# start, before this module is imported by gunicorn.

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

COLLECTION_NAME = "legal_knowledge"

# Uploaded PDFs get their own collection so they never mix into the
# main legal_knowledge results for other users' default (no-document)
# questions. Every chunk stored here carries a "session_id" in its
# metadata, and every read/delete against this collection is filtered
# by that session_id, so one browser tab only ever sees its own upload.
UPLOADS_COLLECTION_NAME = "user_uploaded_documents"

MISTRAL_MODEL = "mistral-large-latest"


# -------------------------------------------------------
# Validate Environment Variables
# -------------------------------------------------------
# Fail fast and loud on missing config, with a message that points
# straight at the fix (Render dashboard -> Environment), instead of
# a bare Exception that just says a name is missing.

_missing_env_vars = [
    name for name, value in [
        ("QDRANT_URL", QDRANT_URL),
        ("QDRANT_API_KEY", QDRANT_API_KEY),
        ("MISTRAL_API_KEY", MISTRAL_API_KEY),
    ] if not value
]

if _missing_env_vars:
    print(
        "STARTUP FAILED - missing required environment variable(s): "
        + ", ".join(_missing_env_vars)
    )
    print(
        "Fix: Render dashboard -> your service -> Environment tab -> "
        "add each key above with its real value, then redeploy."
    )
    sys.exit(1)


# -------------------------------------------------------
# Load Embedding Model
# -------------------------------------------------------
# NOTE: this downloads/loads the all-MiniLM-L6-v2 weights at process
# start, before gunicorn opens the port. On a cold instance this can
# take a noticeable amount of time and Render's "No open ports
# detected, continuing to scan..." log line is expected while it's
# happening - it isn't an error by itself.

print("Loading embedding model...")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
except Exception as e:
    print(f"STARTUP FAILED - could not load embedding model: {e}")
    sys.exit(1)

print("Embedding model loaded.")


# -------------------------------------------------------
# Connect Qdrant
# -------------------------------------------------------

print("Connecting to Qdrant...")

try:
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
except Exception as e:
    print(f"STARTUP FAILED - could not connect to Qdrant collection "
          f"'{COLLECTION_NAME}': {e}")
    print(
        "Fix: confirm QDRANT_URL/QDRANT_API_KEY point at the same "
        "cluster you originally uploaded chunks to from Colab, and "
        "that the 'legal_knowledge' collection exists there."
    )
    sys.exit(1)

print("Qdrant connected.")


# -------------------------------------------------------
# Uploads Collection (per-user-session PDF chunks)
# -------------------------------------------------------
# A single shared QdrantClient is used both for the low-level
# collection-management calls below (create/delete) and, wrapped in
# QdrantVectorStore, for the langchain add_documents/similarity_search
# calls used in /upload and /chat.

qdrant_client_instance = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

existing_collection_names = [
    collection.name
    for collection in qdrant_client_instance.get_collections().collections
]

if UPLOADS_COLLECTION_NAME not in existing_collection_names:

    print(f"Creating '{UPLOADS_COLLECTION_NAME}' collection for uploaded PDFs...")

    embedding_dimension = len(embeddings.embed_query("dimension probe"))

    qdrant_client_instance.create_collection(
        collection_name=UPLOADS_COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=embedding_dimension,
            distance=qdrant_models.Distance.COSINE
        )
    )

    print(f"'{UPLOADS_COLLECTION_NAME}' collection created.")

uploads_vector_store = QdrantVectorStore(
    client=qdrant_client_instance,
    collection_name=UPLOADS_COLLECTION_NAME,
    embedding=embeddings
)


# -------------------------------------------------------
# Mistral Client
# -------------------------------------------------------

print("Connecting to Mistral...")

try:
    mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
except Exception as e:
    print(f"STARTUP FAILED - could not create Mistral client: {e}")
    sys.exit(1)

print("Mistral connected.")


# -------------------------------------------------------
# Speech Recognizer (for voice input)
# -------------------------------------------------------
# dynamic_energy_threshold lets the recognizer adapt its
# silence/speech cutoff per-clip instead of using one fixed
# threshold for every recording. This alone fixes a large
# share of "Could not understand the audio" false negatives
# caused by quieter mics or background noise.

recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
recognizer.energy_threshold = 300  # sane starting point, auto-adjusted below


# -------------------------------------------------------
# Uploaded Document Settings
# -------------------------------------------------------

MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB

MIN_AUDIO_DURATION_MS = 400  # clips shorter than this are almost never valid speech

# Lightweight lookup so /chat and /remove-document can check "does this
# session have an active uploaded document" without hitting Qdrant.
# The actual chunk vectors live in the UPLOADS_COLLECTION_NAME
# collection in Qdrant, tagged with metadata.session_id — this dict is
# just a fast local index of which session_ids are currently active.
active_upload_sessions = {}
# structure: { session_id: {"filename": str, "chunks_indexed": int} }


# -------------------------------------------------------
# HTML Page
# -------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------------------------
# Health Check
# -------------------------------------------------------
# A cheap endpoint that doesn't touch Qdrant/Mistral - useful for
# Render health checks or uptime monitors, and as a fast way to
# confirm the process is actually up and serving requests.

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# -------------------------------------------------------
# Upload API
# -------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():

    try:

        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file was uploaded."
            }), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return jsonify({
                "status": "error",
                "message": "No file was selected."
            }), 400

        if not uploaded_file.filename.lower().endswith(".pdf"):
            return jsonify({
                "status": "error",
                "message": "Only PDF files are supported."
            }), 400

        file_bytes = uploaded_file.read()

        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            return jsonify({
                "status": "error",
                "message": "File is too large. Maximum size is 15 MB."
            }), 400

        reader = PdfReader(io.BytesIO(file_bytes))

        session_id = str(uuid.uuid4())

        raw_documents = []

        for page_index, page in enumerate(reader.pages):

            page_text = page.extract_text() or ""
            page_text = page_text.strip()

            if page_text == "":
                continue

            raw_documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": uploaded_file.filename,
                        "page": page_index,
                        "session_id": session_id
                    }
                )
            )

        if len(raw_documents) == 0:
            return jsonify({
                "status": "error",
                "message": "Could not extract any readable text from this PDF."
            }), 400

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunked_documents = splitter.split_documents(raw_documents)

        # Every chunk keeps the session_id metadata set above (the
        # splitter carries metadata through), so it can be filtered
        # and later deleted by session_id.
        uploads_vector_store.add_documents(chunked_documents)

        active_upload_sessions[session_id] = {
            "filename": uploaded_file.filename,
            "chunks_indexed": len(chunked_documents)
        }

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "filename": uploaded_file.filename,
            "chunks_indexed": len(chunked_documents)
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# -------------------------------------------------------
# Remove Uploaded Document
# -------------------------------------------------------

@app.route("/remove-document", methods=["POST"])
def remove_document():
    # Call this whenever a document should stop being usable for
    # future questions: user removes the attachment, or the chat
    # widget is closed / a "new chat" is started on the frontend.
    # It deletes the actual vectors from Qdrant (not just local
    # bookkeeping), so no trace of that session's document remains.

    try:

        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "")

        if session_id == "":
            return jsonify({"status": "success"})

        qdrant_client_instance.delete(
            collection_name=UPLOADS_COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.session_id",
                            match=qdrant_models.MatchValue(value=session_id)
                        )
                    ]
                )
            )
        )

        active_upload_sessions.pop(session_id, None)

        return jsonify({"status": "success"})

    except Exception as e:

        print(f"[remove-document] Failed to clean up session {session_id}: {e}")

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# -------------------------------------------------------
# Voice Transcription API
# -------------------------------------------------------
# RENDER DEPLOYMENT NOTE: this endpoint decodes audio via pydub, which
# shells out to the "ffmpeg" binary. Render's native Python runtime
# does NOT include ffmpeg, so on a plain Python web service this route
# will fail at the AudioSegment.from_file() call below (caught and
# returned as a 400 "Could not decode the audio recording" - it will
# not crash the whole app, only the /transcribe feature). To make
# voice input work on Render you need to deploy via a Dockerfile that
# installs ffmpeg (e.g. `apt-get install -y ffmpeg` in the image)
# instead of the native Python runtime.
#
# Changes vs. the original version:
#
# 1. Ambient-noise calibration: recognizer.adjust_for_ambient_noise()
#    is run against the first fraction of a second of the clip so the
#    energy threshold adapts to that specific recording instead of
#    using one static value for every user/mic.
#
# 2. Volume normalization: pydub's normalize() boosts quiet clips up
#    to a consistent peak level before recognition. Quiet mic input
#    was one of the most common causes of UnknownValueError.
#
# 3. Duration / silence guard: clips under MIN_AUDIO_DURATION_MS or
#    clips that are near-silent (very low dBFS) are rejected with a
#    specific, actionable message instead of the generic
#    "could not understand" error, so you can tell from the response
#    itself whether the mic/recorder is the problem.
#
# 4. Server-side logging of failures: prints the exception/detail to
#    the server console (not to the client) so this is debuggable in
#    production logs.

@app.route("/transcribe", methods=["POST"])
def transcribe():

    input_audio_path = None
    wav_audio_path = None

    try:

        if "audio" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No audio was received."
            }), 400

        audio_file = request.files["audio"]

        audio_bytes = audio_file.read()

        if len(audio_bytes) == 0:
            return jsonify({
                "status": "error",
                "message": "Empty audio recording."
            }), 400

        if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
            return jsonify({
                "status": "error",
                "message": "Audio recording is too large."
            }), 400

        # ----------------------------------
        # Save incoming audio to a temp file
        # ----------------------------------

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as input_temp:
            input_temp.write(audio_bytes)
            input_audio_path = input_temp.name

        # ----------------------------------
        # Decode with ffmpeg (via pydub). Format is
        # auto-detected from file content, not extension,
        # so this works regardless of which codec the
        # browser's MediaRecorder actually used.
        # ----------------------------------

        try:
            audio_segment = AudioSegment.from_file(input_audio_path)
        except Exception as decode_error:
            print(f"[transcribe] ffmpeg decode failed: {decode_error}")
            return jsonify({
                "status": "error",
                "message": "Could not decode the audio recording. Please try again."
            }), 400

        # ----------------------------------
        # Reject clips that are too short to contain speech
        # ----------------------------------

        if len(audio_segment) < MIN_AUDIO_DURATION_MS:
            return jsonify({
                "status": "error",
                "message": "Recording was too short. Please hold the mic button and speak for at least a second."
            }), 400

        # ----------------------------------
        # Normalize volume, then check for near-silence
        # ----------------------------------

        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_segment = pydub_normalize(audio_segment)

        if audio_segment.dBFS == float("-inf") or audio_segment.dBFS < -45:
            return jsonify({
                "status": "error",
                "message": "Recording seems silent. Please check your microphone permissions and try again."
            }), 400

        wav_audio_path = input_audio_path + ".wav"
        audio_segment.export(wav_audio_path, format="wav")

        # ----------------------------------
        # Transcribe using SpeechRecognition
        # ----------------------------------

        with sr.AudioFile(wav_audio_path) as source:
            # Calibrate to this specific clip's noise floor before
            # recording, instead of relying on a single fixed
            # energy_threshold for every request.
            recognizer.adjust_for_ambient_noise(source, duration=min(0.5, source.DURATION))
            audio_data = recognizer.record(source)

        try:
            transcript = recognizer.recognize_google(audio_data, language="en-IN")
        except sr.UnknownValueError:
            print("[transcribe] recognize_google: UnknownValueError (no speech detected in clip)")
            return jsonify({
                "status": "error",
                "message": "Could not understand the audio. Please speak clearly and try again."
            }), 400
        except sr.RequestError as e:
            print(f"[transcribe] recognize_google: RequestError: {e}")
            return jsonify({
                "status": "error",
                "message": f"Speech recognition service error: {str(e)}"
            }), 503

        return jsonify({
            "status": "success",
            "transcript": transcript
        })

    except Exception as e:

        print(f"[transcribe] Unhandled error: {e}")

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:

        for path in (input_audio_path, wav_audio_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# -------------------------------------------------------
# Chat API
# -------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(silent=True) or {}

        question = data.get("question", "")
        question = question.strip() if isinstance(question, str) else ""

        session_id = data.get("session_id", "")

        if question == "":
            return jsonify({
                "status": "error",
                "message": "Question is required."
            }), 400

        using_uploaded_document = session_id in active_upload_sessions

        if using_uploaded_document:

            # Filter restricts the search to chunks tagged with this
            # exact session_id, so only the PDF this session uploaded
            # can ever be retrieved here — not other sessions' uploads
            # and not the main legal_knowledge collection.
            session_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.session_id",
                        match=qdrant_models.MatchValue(value=session_id)
                    )
                ]
            )

            docs = uploads_vector_store.similarity_search(
                question,
                k=5,
                filter=session_filter
            )

        else:

            docs = vector_store.similarity_search(
                question,
                k=5
            )

        context_blocks = []

        for doc in docs:

            source_name = doc.metadata.get("source", "Unknown Document")
            page_number = doc.metadata.get("page", None)

            if page_number is not None:
                citation_label = f"[Source: {source_name} | Page {int(page_number) + 1}]"
            else:
                citation_label = f"[Source: {source_name}]"

            context_blocks.append(f"{citation_label}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
You are Legal Knowledge Twin.

You are an expert AI legal assistant for Indian law.

Answer ONLY from the supplied legal documents below. Do not hallucinate
and do not use any outside knowledge.

Each context block below is tagged with its exact source document name
and page number in the format [Source: filename | Page N]. When you cite
a source in your answer, copy that filename and page number exactly as
given. Never invent, guess, or alter a source name or page number. If a
context block has no page number listed, cite only the filename.

Context:

{context}

User Question:

{question}

Formatting rules (follow strictly):
- Plain text only. Do not use markdown symbols such as asterisks (*),
  underscores (_), backticks, or hash symbols (#) anywhere in the answer.
- Do not bold, italicize, or otherwise stylize any words.
- Use plain section headers exactly as written below, followed by a
  colon and a line break.

Respond using this exact format:

Summary:
<short plain-text answer>

Explanation:
<clear plain-text explanation>

Sources:
<exact filename and page number for each source actually used, one per line>

What should the user do next:
<practical suggestion>

If the answer is not available in the context, reply exactly:

Sorry, I could not find this information in the uploaded legal documents.
"""

        response = mistral_client.chat(
            model=MISTRAL_MODEL,
            messages=[
                ChatMessage(role="user", content=prompt)
            ]
        )

        answer = response.choices[0].message.content

        for symbol in ["**", "__", "`", "##", "#"]:
            answer = answer.replace(symbol, "")

        answer = answer.replace("*", "")

        sources = []

        for doc in docs:

            source = doc.metadata.get("source", "Unknown")

            if source not in sources:
                sources.append(source)

        return jsonify({

            "status": "success",

            "question": question,

            "answer": answer,

            "sources": sources,

            "source_mode": "uploaded_document" if using_uploaded_document else "knowledge_base"

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# -------------------------------------------------------
# Run App
# -------------------------------------------------------

if __name__ == "__main__":
    # This block only runs when you execute `python app.py` directly
    # (e.g. local testing). On Render, gunicorn imports this module
    # and calls the `app` object directly - this block is skipped,
    # and the actual bind/port/host come from your Render Start
    # Command instead, e.g.:
    #   gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
    # The --timeout 120 is worth keeping in the Start Command since
    # the first request after a cold start can be slow while the
    # embedding model finishes warming up.

    PORT = int(os.environ.get("PORT", 5000))

    print(f"Server running on port {PORT}")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
