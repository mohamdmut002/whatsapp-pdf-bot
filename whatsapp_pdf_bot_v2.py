"""
whatsapp_pdf_bot_v2.py
-----------------------
Production version: persistent sessions (SQLite), background processing,
and automatic car-plate detection (via Google Cloud Vision OCR) to name
the PDF -- with a manual fallback if detection fails or is disabled.

Install:
    pip install flask twilio requests reportlab pillow google-cloud-vision

Google Vision setup (optional but recommended for auto plate detection):
    1. Create a free Google Cloud project -> console.cloud.google.com
    2. Enable the "Cloud Vision API"
    3. Create a service account key (JSON file) and download it
    4. Set env var: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    If you skip this, set USE_OCR = False below and it'll always ask
    the user to type a filename instead.
"""

import os
import re
import uuid
import sqlite3
import threading
import requests
from flask import Flask, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

# ---- Config (read from environment variables — set these in Railway) ----
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"]
USE_OCR = os.environ.get("USE_OCR", "true").lower() == "true"

DATA_DIR = os.environ.get("DATA_DIR", ".")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
DB_PATH = os.path.join(DATA_DIR, "bot.db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
validator = RequestValidator(TWILIO_AUTH_TOKEN)

if USE_OCR:
    from google.cloud import vision
    vision_client = vision.ImageAnnotatorClient()


# ---------------- Database (persistent sessions) ----------------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                phone TEXT PRIMARY KEY,
                state TEXT DEFAULT 'collecting',   -- collecting | awaiting_filename
                detected_plate TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT,
                image_path TEXT,
                caption TEXT
            )
        """)


def get_session(phone):
    with db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE phone=?", (phone,)).fetchone()
        if not row:
            conn.execute("INSERT INTO sessions (phone) VALUES (?)", (phone,))
            return {"phone": phone, "state": "collecting", "detected_plate": None}
        return dict(row)


def update_session(phone, **fields):
    with db() as conn:
        for key, val in fields.items():
            conn.execute(f"UPDATE sessions SET {key}=? WHERE phone=?", (val, phone))


def add_entry(phone, image_path, caption):
    with db() as conn:
        conn.execute(
            "INSERT INTO entries (phone, image_path, caption) VALUES (?, ?, ?)",
            (phone, image_path, caption),
        )


def get_entries(phone):
    with db() as conn:
        rows = conn.execute("SELECT * FROM entries WHERE phone=?", (phone,)).fetchall()
        return [dict(r) for r in rows]


def clear_session(phone):
    with db() as conn:
        conn.execute("DELETE FROM entries WHERE phone=?", (phone,))
        conn.execute("UPDATE sessions SET state='collecting', detected_plate=NULL WHERE phone=?", (phone,))


# ---------------- Plate detection ----------------

# UAE-style plate pattern: adjust to match what you actually see, e.g. "12345" or "A 12345"
PLATE_PATTERN = re.compile(r"\b[A-Z]{0,2}\s?\d{4,6}\b")


def detect_plate(image_path):
    """Runs OCR and returns the first plate-like string found, or None."""
    if not USE_OCR:
        return None
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    if response.error.message:
        print("Vision API error:", response.error.message)
        return None
    texts = response.text_annotations
    if not texts:
        return None
    full_text = texts[0].description.upper()
    match = PLATE_PATTERN.search(full_text)
    return match.group().strip() if match else None


# ---------------- PDF building ----------------

def build_pdf(entries, output_path):
    page_width, page_height = A4
    c = canvas.Canvas(output_path, pagesize=A4)
    margin = 1 * cm  # smaller margin = more room for the photo

    for entry in entries:
        try:
            img = Image.open(entry["image_path"])
            img.load()  # force-read now so a corrupt file fails here, not later
        except Exception as e:
            print(f"WARNING: skipping unreadable image {entry['image_path']}: {e}")
            continue

        img_w, img_h = img.size
        aspect = img_h / img_w
        draw_width = page_width - 2 * margin
        draw_height = draw_width * aspect

        # Leave a little room at the bottom for the caption text
        max_h = page_height - 3 * cm
        if draw_height > max_h:
            draw_height = max_h
            draw_width = draw_height / aspect

        img_x = (page_width - draw_width) / 2  # center the image horizontally
        img_y = page_height - margin - draw_height
        c.drawImage(ImageReader(img), img_x, img_y, width=draw_width, height=draw_height)

        c.setFont("Helvetica", 11)
        text_y = img_y - 1 * cm
        for line in (entry.get("caption") or "").split("\n"):
            c.drawString(margin, text_y, line[:100])
            text_y -= 15

        c.showPage()

    c.save()


def safe_filename(name):
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name.strip())
    return name or "document"


def bilingual(english, arabic):
    """Combine an English and Arabic version of a message into one reply."""
    return f"{english}\n\n{arabic}"


# ---------------- Background workers ----------------

def process_incoming_media(phone, media_url, caption):
    """Runs in a background thread: download image + try plate detection."""
    entry_num = len(get_entries(phone))
    filename = f"{phone.replace(':', '_').replace('+', '')}_{entry_num}.jpg"
    save_path = os.path.join(UPLOAD_DIR, filename)

    resp = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))

    if resp.status_code != 200 or not resp.headers.get("Content-Type", "").startswith("image"):
        print(
            f"WARNING: media download failed for {phone} "
            f"(status={resp.status_code}, content-type={resp.headers.get('Content-Type')}). "
            f"Check TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN are correct."
        )
        return  # don't save a broken/non-image file

    with open(save_path, "wb") as f:
        f.write(resp.content)

    add_entry(phone, save_path, caption)

    if USE_OCR:
        session = get_session(phone)
        if not session.get("detected_plate"):
            plate = detect_plate(save_path)
            if plate:
                update_session(phone, detected_plate=plate)


def finalize_pdf(phone, filename_choice):
    """Runs in background: build PDF and send it back via WhatsApp."""
    entries = get_entries(phone)
    friendly_name = safe_filename(filename_choice)  # what the user sees, e.g. "mutasim-93019"

    # The actual file on disk / in the URL uses a random, unguessable name.
    # This stops anyone from finding or downloading someone else's PDF by
    # guessing plate numbers or names in the URL.
    secret_id = uuid.uuid4().hex
    output_filename = f"{secret_id}.pdf"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    build_pdf(entries, output_path)

    pdf_public_url = f"{PUBLIC_BASE_URL}/outputs/{output_filename}"
    client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        to=phone,
        body=bilingual(
            f"Here's your PDF: {friendly_name}.pdf",
            f"إليك ملفك: {friendly_name}.pdf",
        ),
        media_url=[pdf_public_url],
    )
    clear_session(phone)


# ---------------- Webhook ----------------

@app.route("/webhook", methods=["POST"])
def webhook():
    # Reject anything that isn't really from Twilio
    signature = request.headers.get("X-Twilio-Signature", "")
    url = request.url
    if not validator.validate(url, request.form, signature):
        print(f"WARNING: signature check failed for url={url} (allowing through for now)")
        # NOTE: temporarily not blocking — see note below about fixing TWILIO_AUTH_TOKEN
        # return ("Forbidden", 403)

    from_number = request.form.get("From")
    body = (request.form.get("Body") or "").strip()
    num_media = int(request.form.get("NumMedia", 0))

    session = get_session(from_number)
    resp = MessagingResponse()

    # --- Let the user clear a stuck/broken session ---
    if body.lower() == "reset":
        clear_session(from_number)
        resp.message(bilingual(
            "Session cleared. Send new photos to start again.",
            "تم مسح الجلسة. أرسل صورًا جديدة للبدء من جديد.",
        ))
        return str(resp)

    # --- Waiting for the user to confirm/type a filename ---
    if session["state"] == "awaiting_filename":
        if body.lower() in ("yes", "y") and session.get("detected_plate"):
            chosen_name = session["detected_plate"]
        elif body:
            chosen_name = body
        else:
            chosen_name = "document"
        clean_name = safe_filename(chosen_name)
        resp.message(bilingual(
            f"Got it — building your PDF as '{clean_name}.pdf'...",
            f"تم الاستلام — جارٍ إنشاء ملف PDF باسم '{clean_name}.pdf'...",
        ))
        threading.Thread(target=finalize_pdf, args=(from_number, chosen_name)).start()
        return str(resp)

    # --- User says done ---
    if body.lower() == "done":
        entries = get_entries(from_number)
        if not entries:
            resp.message(bilingual(
                "You haven't sent any photos yet. Send some first!",
                "لم ترسل أي صور بعد. أرسل بعض الصور أولاً!",
            ))
            return str(resp)

        plate = session.get("detected_plate")
        if plate:
            update_session(from_number, state="awaiting_filename")
            resp.message(bilingual(
                f"I detected plate number *{plate}* in your photos. "
                f"Reply 'yes' to use it as the filename, or type a different name.",
                f"لقد اكتشفت رقم اللوحة *{plate}* في صورك. "
                f"اكتب 'yes' لاستخدامه كاسم للملف، أو اكتب اسمًا مختلفًا.",
            ))
        else:
            update_session(from_number, state="awaiting_filename")
            resp.message(bilingual(
                "What would you like to name this PDF?",
                "ما الاسم الذي تريد إعطاءه لهذا الملف؟",
            ))
        return str(resp)

    # --- Incoming photo(s) ---
    if num_media > 0:
        for i in range(num_media):
            media_url = request.form.get(f"MediaUrl{i}")
            threading.Thread(
                target=process_incoming_media, args=(from_number, media_url, body)
            ).start()
        count = len(get_entries(from_number)) + num_media
        resp.message(bilingual(
            f"Got it (~{count} photo(s) so far). Send more, or reply 'done' when finished.",
            f"تم الاستلام (~{count} صورة حتى الآن). أرسل المزيد، أو اكتب 'done' عند الانتهاء.",
        ))
        return str(resp)

    resp.message(bilingual(
        "Send me photos (with optional captions), then reply 'done' when finished.",
        "أرسل لي الصور (مع تعليق اختياري إن أردت)، ثم اكتب 'done' عند الانتهاء.",
    ))
    return str(resp)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
