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
# All credentials come from environment variables (set on Render)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GROQ_API_KEY:
    raise RuntimeError("Missing required environment variables")

# Global client (anonymous – only for auth validation)
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

Rules:
- amount_paid: the dollar amount the claimant HAS RECEIVED (e.g., "10000"). If multiple numbers, pick the one labeled "received" or "amount paid".
- amount_due: the dollar amount the claimant IS STILL OWED (e.g., "2345"). Look for phrases like "amount left to be paid", "remaining", "still due". Do NOT compute subtraction. Return the number as a string without commas or currency symbol.
- If you see a range or subtraction, ignore it. Only extract explicit standalone numbers.
- payment_date: format YYYY-MM-DD.
- waiver_type: "partial", "final", "conditional", or "unconditional".

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
    result = json.loads(completion.choices[0].message.content)

    # Post-process numeric fields: extract first number from any string
    import re
    def extract_number(val):
        if not val or not isinstance(val, str):
            return val
        # Find first sequence of digits (with optional decimal)
        match = re.search(r'\d+(?:\.\d+)?', val)
        return match.group(0) if match else ''
    
    result['amount_paid'] = extract_number(result.get('amount_paid', ''))
    result['amount_due'] = extract_number(result.get('amount_due', ''))
    return result

# ------------------- Authentication -------------------
def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token = authorization.replace("Bearer ", "")
    try:
        user = supabase_anon.auth.get_user(token)
        return token, user.user.id  # returns tuple (token, user_id)
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

    # Skip storage upload for speed (optional)
    pdf_url = ""

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
        raise HTTPException(status_code=500, detail="Failed to save extraction results")

    return {"status": "completed", "data": extracted}

# ------------------- API Endpoints -------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    auth_data: tuple = Depends(get_current_user)
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