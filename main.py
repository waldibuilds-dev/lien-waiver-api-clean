import os
import uuid
import tempfile
import re
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from groq import Groq
import PyPDF2

# ========== CONFIGURATION ==========
# REPLACE THESE with your actual keys from Supabase and Groq
SUPABASE_URL = "https://yataoyjfxacqnfgsmaeg.supabase.co"      # <-- CHANGE
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlhdGFveWpmeGFjcW5mZ3NtYWVnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA3NTU0NzcsImV4cCI6MjA5NjMzMTQ3N30.9NiseMRBGbq80FiFYIFTIDnn0haed8A_IlPQZwcoXMI"                # <-- CHANGE
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# Global clients (anon)
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

def process_upload(access_token: str, user_id: str, file_content: bytes, filename: str):
    # Create an authenticated Supabase client using the user's token
    supabase_auth: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase_auth.auth.set_session(access_token, access_token)

    print(f"Processing upload for user {user_id}, file {filename}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        extracted = run_extraction(tmp_path)
        print(f"Extraction result: {extracted}")
    except Exception as e:
        print(f"Extraction error: {e}")
        extracted = {"error": str(e)}
    finally:
        os.unlink(tmp_path)

    # Optional storage upload (requires bucket and policies)
    pdf_url = ""
    bucket_name = "lien-waivers"
    try:
        file_path = f"{user_id}/{uuid.uuid4()}.pdf"
        supabase_auth.storage.from_(bucket_name).upload(file_path, file_content)
        pdf_url = supabase_auth.storage.from_(bucket_name).get_public_url(file_path)
        print(f"Uploaded to storage: {pdf_url}")
    except Exception as e:
        print(f"Storage upload error: {e}")

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
        print("Database insert successful")
    except Exception as e:
        print(f"Database insert error: {e}")

def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token = authorization.replace("Bearer ", "")
    try:
        # Verify token and get user
        user = supabase_anon.auth.get_user(token)
        return token, user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auth_data: tuple = Depends(get_current_user)
):
    access_token, user_id = auth_data
    contents = await file.read()
    background_tasks.add_task(process_upload, access_token, user_id, contents, file.filename)
    return {"status": "processing", "message": "Your waiver is being processed"}

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