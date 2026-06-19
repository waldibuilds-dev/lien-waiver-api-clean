import os
import uuid
import tempfile
import re
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from groq import Groq
import PyPDF2
import stripe

# ========== CONFIGURATION ==========
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

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
    def extract_number(val):
        if not val or not isinstance(val, str):
            return val
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
    
    # Check subscription status (allow active or trialing)
    supabase_auth = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase_auth.auth.set_session(access_token, access_token)
    sub_result = supabase_auth.table("subscriptions").select("*").eq("user_id", user_id).in_("status", ["active", "trialing"]).execute()
    if not sub_result.data:
        raise HTTPException(status_code=402, detail="Active or trial subscription required. Please subscribe.")
    
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

@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    event_data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        session = event_data
        customer_email = session["customer_details"]["email"]
        stripe_customer_id = session["customer"]
        stripe_subscription_id = session["subscription"]
        supabase_user_id = session.get("metadata", {}).get("supabase_user_id")

        if not supabase_user_id:
            supabase_admin = create_client(SUPABASE_URL, os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
            user_result = supabase_admin.auth.admin.list_users()
            for user in user_result.users:
                if user.email == customer_email:
                    supabase_user_id = user.id
                    break

        if not supabase_user_id:
            print("Webhook error: no user_id found")
            return {"status": "error", "detail": "User not found"}

        try:
            subscription = stripe.Subscription.retrieve(stripe_subscription_id)
            status = subscription.status
            current_period_end = datetime.fromtimestamp(subscription.current_period_end).isoformat()
        except Exception as e:
            print(f"Error retrieving subscription: {e}")
            status = "trialing"
            current_period_end = (datetime.now() + timedelta(days=30)).isoformat()

        supabase_auth = create_client(SUPABASE_URL, SUPABASE_KEY)
        data = {
            "user_id": supabase_user_id,
            "stripe_customer_id": stripe_customer_id,
            "stripe_subscription_id": stripe_subscription_id,
            "status": status,
            "current_period_end": current_period_end,
            "updated_at": datetime.now().isoformat(),
        }
        existing = supabase_auth.table("subscriptions").select("*").eq("user_id", supabase_user_id).execute()
        if existing.data:
            supabase_auth.table("subscriptions").update(data).eq("user_id", supabase_user_id).execute()
        else:
            supabase_auth.table("subscriptions").insert(data).execute()

    elif event_type == "customer.subscription.updated":
        subscription = event_data
        stripe_subscription_id = subscription.id
        status = subscription.status
        current_period_end = datetime.fromtimestamp(subscription.current_period_end).isoformat()

        supabase_admin = create_client(SUPABASE_URL, os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
        result = supabase_admin.table("subscriptions").select("*").eq("stripe_subscription_id", stripe_subscription_id).execute()
        if result.data:
            user_id = result.data[0]["user_id"]
            supabase_auth = create_client(SUPABASE_URL, SUPABASE_KEY)
            supabase_auth.table("subscriptions").update({
                "status": status,
                "current_period_end": current_period_end,
                "updated_at": datetime.now().isoformat()
            }).eq("user_id", user_id).execute()
        else:
            print(f"Webhook warning: subscription {stripe_subscription_id} not found in database")

    return {"status": "ok"}

@app.post("/create-checkout-session")
async def create_checkout_session(auth_data: tuple = Depends(get_current_user)):
    access_token, user_id = auth_data

    supabase_auth = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase_auth.auth.set_session(access_token, access_token)
    user = supabase_auth.auth.get_user()
    email = user.user.email

    # Abuse prevention: check if this user already had a subscription
    supabase_admin = create_client(SUPABASE_URL, os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
    existing = supabase_admin.table("subscriptions").select("*").eq("user_id", user_id).execute()
    if existing.data:
        for sub in existing.data:
            if sub.get("status") in ["active", "trialing", "past_due"]:
                raise HTTPException(
                    status_code=403,
                    detail="You've already used a free trial or have an active subscription. Contact support."
                )

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": "price_1Timcx2H40FY3BJebX8FDbXV",
                "quantity": 1,
            }],
            mode="subscription",
            subscription_data={
                "trial_period_days": 7
            },
            success_url="https://lienflow-frontend.onrender.com?success=true",
            cancel_url="https://lienflow-frontend.onrender.com?canceled=true",
            customer_email=email,
            metadata={"supabase_user_id": user_id},
        )
        return {"url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}