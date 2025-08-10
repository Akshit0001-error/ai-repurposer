"""
AI Content Repurposer - Backend (FastAPI)

Single-file MVP backend that implements:
 - /health                 GET
 - /signup                 POST  (simple email/password -> SQLite)
 - /login                  POST  (returns JWT token)
 - /upload                 POST  (file upload OR YouTube URL)
 - /transcribe/{job_id}    GET   (check status & result)
 - /generate/{job_id}      POST  (generate text outputs from transcription)
 - /download/{job_id}      GET   (zip and download all generated artifacts)

Notes:
 - Uses OpenAI for transcription + text generation (set OPENAI_API_KEY env var)
 - Uses yt-dlp to download audio from YouTube URLs (optional; install it)
 - Uses SQLite (SQLModel) for persistence (easy to replace with Supabase)
 - JWT auth for protected endpoints
 - Background tasks handle long-running jobs (transcription + generation)

Dependencies (pip):
fastapi uvicorn[standard] python-multipart openai pydantic sqlalchemy sqlmodel
python-dotenv yt-dlp aiofiles python-jose[cryptography] passlib[bcrypt]

Run:
 1. pip install -r requirements.txt  (see deps above)
 2. export OPENAI_API_KEY=sk-...   (or create a .env file)
 3. uvicorn ai_repurposer_backend:app --reload --host 0.0.0.0 --port 8000

This file is intentionally compact and documented. Replace/extend pieces (e.g. DB, storage, Stripe) as needed.
"""

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import Optional
import os
import uuid
import shutil
import subprocess
import aiofiles
import time
import zipfile
import io
import openai
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
openai.api_key = OPENAI_API_KEY

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 3600 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR,'db.sqlite') }"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

app = FastAPI(title="AI Content Repurposer - Backend")

# -----------------
# Database models
# -----------------
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str
    hashed_password: str

class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = None
    status: str = "created"  # created, downloading, transcribing, generating, done, failed
    source_type: Optional[str] = None  # "file" or "youtube"
    original_filename: Optional[str] = None
    audio_path: Optional[str] = None
    transcript: Optional[str] = None
    output_json: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())

SQLModel.metadata.create_all(engine)

# -----------------
# Utility functions
# -----------------

def get_db():
    with Session(engine) as session:
        yield session

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[int] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = int(time.time()) + expires_delta
    else:
        expire = int(time.time()) + ACCESS_TOKEN_EXPIRE_SECONDS
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def save_upload_file_tmp(upload_file: UploadFile, destination: str) -> None:
    async with aiofiles.open(destination, 'wb') as out_file:
        content = await upload_file.read()  # async read
        await out_file.write(content)

# -----------------
# Auth dependencies
# -----------------

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        raise credentials_exception
    return user

# -----------------
# Schemas
# -----------------
class SignUpIn(BaseModel):
    email: str
    password: str

class JobOut(BaseModel):
    id: int
    status: str
    transcript: Optional[str] = None

# -----------------
# Auth endpoints
# -----------------
@app.post("/signup")
def signup(payload: SignUpIn, db: Session = Depends(get_db)):
    existing = db.exec(select(User).where(User.email == payload.email)).first()
    if existing:
        return JSONResponse({"error": "email already registered"}, status_code=400)
    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.exec(select(User).where(User.email == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}

# -----------------
# Helper: download YouTube audio via yt-dlp (if user provides URL)
# -----------------

def download_youtube_audio(url: str, dest_folder: str) -> str:
    """Downloads audio from YouTube URL and returns path to audio file (mp3)
    Requires yt-dlp installed in the environment."""
    filename = f"{uuid.uuid4()}.mp3"
    out_path = os.path.join(dest_folder, filename)
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        out_path,
        url
    ]
    # run and wait
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("yt-dlp failed: " + proc.stderr.decode('utf-8'))
    # yt-dlp will replace placeholders; attempt to find the file in the folder
    # For simplicity, assume out_path exists
    if not os.path.exists(out_path):
        # try to find any mp3 in folder modified recently
        files = [os.path.join(dest_folder, f) for f in os.listdir(dest_folder) if f.endswith('.mp3')]
        if not files:
            raise RuntimeError("No audio file found after yt-dlp run")
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]
    return out_path

# -----------------
# Background job: transcribe using OpenAI Whisper + generate outputs
# -----------------

def run_transcribe_and_generate(job_id: int):
    with Session(engine) as db:
        job = db.get(Job, job_id)
        if not job:
            return
        try:
            job.status = "transcribing"
            db.add(job); db.commit(); db.refresh(job)

            audio_path = job.audio_path
            if not audio_path or not os.path.exists(audio_path):
                raise RuntimeError("audio missing")

            # Call OpenAI Whisper (file-based)
            # Using openai.Audio.transcribe (depends on openai package version)
            with open(audio_path, "rb") as af:
                transcript_resp = openai.Audio.transcriptions.create(
                    file=af,
                    model="gpt-4o-mini-transcribe" if hasattr(openai, 'Audio') else "whisper-1",
                )
                # fallback: newer openai bindings vary; the structure below expects a 'text' field
                transcript_text = transcript_resp.get("text") if isinstance(transcript_resp, dict) else getattr(transcript_resp, 'text', None)

            if not transcript_text:
                # try older response shape
                transcript_text = transcript_resp
            job.transcript = transcript_text
            job.status = "generating"
            db.add(job); db.commit(); db.refresh(job)

            # Generate multiple formats using GPT
            prompt = (
                f"You are an assistant that converts a transcript into:\n"
                f"1) A long-form blog draft with headings (approx 700-1000 words).\n"
                f"2) A concise LinkedIn post (4-8 sentences).\n"
                f"3) 6 Tweet-sized posts with hooks and hashtags.\n"
                f"4) 5 short Instagram/TikTok captions.\n"
                f"Here is the transcript:\n\n{transcript_text}\n\nReturn a JSON object with keys: blog, linkedin, tweets (array), captions (array)."
            )
            # Use ChatCompletion or Responses depending on API version
            chat_resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2,
            )
            # extract text
            generated_text = chat_resp["choices"][0]["message"]["content"]
            # store raw output
            job.output_json = generated_text
            job.status = "done"
            db.add(job); db.commit(); db.refresh(job)
        except Exception as e:
            job.status = "failed"
            job.output_json = str(e)
            db.add(job); db.commit(); db.refresh(job)

# -----------------
# API endpoints: upload -> create job -> background transcribe
# -----------------
@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file and not youtube_url:
        raise HTTPException(status_code=400, detail="Provide a file or a youtube_url")

    job = Job(user_id=current_user.id, status="created")
    db.add(job); db.commit(); db.refresh(job)

    job_folder = os.path.join(STORAGE_DIR, str(job.id))
    os.makedirs(job_folder, exist_ok=True)

    try:
        if youtube_url:
            job.source_type = "youtube"
            job.status = "downloading"
            db.add(job); db.commit(); db.refresh(job)
            # download audio sync (could be async)
            audio_path = download_youtube_audio(youtube_url, job_folder)
            job.audio_path = audio_path
        else:
            job.source_type = "file"
            # save uploaded file
            filename = f"{uuid.uuid4()}_{file.filename}"
            dest = os.path.join(job_folder, filename)
            await save_upload_file_tmp(file, dest)
            # if it's video, extract audio using ffmpeg (not implemented here) - assume audio
            job.audio_path = dest
            job.original_filename = file.filename

        job.status = "queued"
        db.add(job); db.commit(); db.refresh(job)

        # schedule background processing
        background_tasks.add_task(run_transcribe_and_generate, job.id)
        return {"job_id": job.id, "status": job.status}
    except Exception as e:
        job.status = "failed"
        job.output_json = str(e)
        db.add(job); db.commit(); db.refresh(job)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut(id=job.id, status=job.status, transcript=job.transcript)

@app.get("/download/{job_id}")
def download_job(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="job not ready for download")

    folder = os.path.join(STORAGE_DIR, str(job.id))
    # create an in-memory zip
    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, mode="w") as zf:
        if job.audio_path and os.path.exists(job.audio_path):
            zf.write(job.audio_path, arcname=os.path.basename(job.audio_path))
        if job.transcript:
            zf.writestr("transcript.txt", job.transcript)
        if job.output_json:
            zf.writestr("outputs.json", job.output_json)
    zip_stream.seek(0)
    return FileResponse(zip_stream, media_type='application/zip', filename=f"job_{job_id}_outputs.zip")

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------
# Simple root message
# -----------------
@app.get("/")
def root():
    return {"message": "AI Content Repurposer Backend. See /docs for interactive API docs."}
