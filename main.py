import os
import uuid
import tempfile
import re
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from groq import Groq
import PyPDF2

# ========== CONFIGURATION ==========
import os

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GROQ_API_KEY:
    raise RuntimeError("Missing required environment variables: SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY")

# Global client (anonymous – only for auth validation and public access)
supabase_anon: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Lien Waiver Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Helper functions -------------------
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    text = re.sub(r'\\\(\\mathbb\{S\}\\\)', '$', text)
    return text

def run_extraction(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    prompt = f"""
Extract the following from this construction lien waiver document. Return ONLY valid JSON.
Fields: claimant_name, customer_name, project_name, owner_name, amount_paid, amount_due, payment_date, waiver_type.
If a field is missing, use empty string.

Document text:
{pdf_text[:12000]}
"""
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)

# ------------------- Authentication -------------------
def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token = authorization.replace("Bearer ", "")
    try:
        user = supabase_anon.auth.get_user(token)
        return token, user.user.id  # returns tuple
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

# ------------------- Synchronous processing -------------------
async def process_upload_sync(access_token: str, user_id: str, file_content: bytes, filename: str):
    # Create an authenticated Supabase client using the user's token
    supabase_auth: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase_auth.auth.set_session(access_token, access_token)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        extracted = run_extraction(tmp_path)
    except Exception as e:
        extracted = {"error": str(e)}
    finally:
        os.unlink(tmp_path)

    # Optional: upload to storage (skip for speed if not needed)
    pdf_url = ""
    bucket_name = "lien-waivers"
    try:
        file_path = f"{user_id}/{uuid.uuid4()}.pdf"
        supabase_auth.storage.from_(bucket_name).upload(file_path, file_content)
        pdf_url = supabase_auth.storage.from_(bucket_name).get_public_url(file_path)
    except Exception as e:
        print(f"Storage upload skipped: {e}")

    data = {
        "user_id": user_id,
        "original_filename": filename,
        "pdf_url": pdf_url,
        "claimant_name": extracted.get("claimant_name", ""),
        "customer_name": extracted.get("customer_name", ""),
        "project_name": extracted.get("project_name", ""),
        "owner_name": extracted.get("owner_name", ""),
        "amount_paid": extracted.get("amount_paid", ""),
        "amount_due": extracted.get("amount_due", ""),
        "payment_date": extracted.get("payment_date", ""),
        "waiver_type": extracted.get("waiver_type", ""),
    }

    try:
        supabase_auth.table("extractions").insert(data).execute()
    except Exception as e:
        print(f"Database insert error: {e}")
        # Re-raise to let the endpoint know
        raise HTTPException(status_code=500, detail="Failed to save extraction results")

    return {"status": "completed", "data": extracted}

# ------------------- API Endpoints -------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    auth_data: tuple = Depends(get_current_user)  # (token, user_id)
):
    access_token, user_id = auth_data
    contents = await file.read()
    result = await process_upload_sync(access_token, user_id, contents, file.filename)
    return result

@app.get("/extractions")
def list_extractions(auth_data: tuple = Depends(get_current_user)):
    access_token, user_id = auth_data
    supabase_auth: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase_auth.auth.set_session(access_token, access_token)
    result = supabase_auth.table("extractions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return result.data

@app.get("/health")
def health():
    return {"status": "ok"}