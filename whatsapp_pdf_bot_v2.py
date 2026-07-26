"""
whatsapp_pdf_bot_v2.py 
-----------------------
Production version: persistent sessions (SQLite), background processing,
and automatic car-plate detection (via Google Cloud Vision OCR) to name
the PDF -- with a manual fallback if detection fails or is disabled.
hi test
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
import sqlite3
import threading
import requests
from flask import Flask, request, send_from_directory
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
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
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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
    margin = 2 * cm

    for entry in entries:
        img = Image.open(entry["image_path"])
        img_w, img_h = img.size
        aspect = img_h / img_w
        draw_width = page_width - 2 * margin
        draw_height = draw_width * aspect
        max_h = 15 * cm
        if draw_height > max_h:
            draw_height = max_h
            draw_width = draw_height / aspect

        img_y = page_height - margin - draw_height
        c.drawImage(ImageReader(img), margin, img_y, width=draw_width, height=draw_height)

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


# ---------------- Background workers ----------------

def process_incoming_media(phone, media_url, caption):
    """Runs in a background thread: download image + try plate detection."""
    entry_num = len(get_entries(phone))
    filename = f"{phone.replace(':', '_').replace('+', '')}_{entry_num}.jpg"
    save_path = os.path.join(UPLOAD_DIR, filename)

    resp = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
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
    safe_name = safe_filename(filename_choice)
    output_filename = f"{safe_name}.pdf"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    build_pdf(entries, output_path)

    pdf_public_url = f"{PUBLIC_BASE_URL}/outputs/{output_filename}"
    client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        to=phone,
        body=f"Here's your PDF: {output_filename}",
        media_url=[pdf_public_url],
    )
    clear_session(phone)


# ---------------- Webhook ----------------

@app.route("/webhook", methods=["POST"])
def webhook():
    from_number = request.form.get("From")
    body = (request.form.get("Body") or "").strip()
    num_media = int(request.form.get("NumMedia", 0))

    session = get_session(from_number)
    resp = MessagingResponse()

    # --- Waiting for the user to confirm/type a filename ---
    if session["state"] == "awaiting_filename":
        if body.lower() in ("yes", "y") and session.get("detected_plate"):
            chosen_name = session["detected_plate"]
        elif body:
            chosen_name = body
        else:
            chosen_name = "document"
        resp.message(f"Got it — building your PDF as '{safe_filename(chosen_name)}.pdf'...")
        threading.Thread(target=finalize_pdf, args=(from_number, chosen_name)).start()
        return str(resp)

    # --- User says done ---
    if body.lower() == "done":
        entries = get_entries(from_number)
        if not entries:
            resp.message("You haven't sent any photos yet. Send some first!")
            return str(resp)

        plate = session.get("detected_plate")
        if plate:
            update_session(from_number, state="awaiting_filename")
            resp.message(
                f"I detected plate number *{plate}* in your photos. "
                f"Reply 'yes' to use it as the filename, or type a different name."
            )
        else:
            update_session(from_number, state="awaiting_filename")
            resp.message("What would you like to name this PDF?")
        return str(resp)

    # --- Incoming photo(s) ---
    if num_media > 0:
        for i in range(num_media):
            media_url = request.form.get(f"MediaUrl{i}")
            threading.Thread(
                target=process_incoming_media, args=(from_number, media_url, body)
            ).start()
        count = len(get_entries(from_number)) + num_media
        resp.message(f"Got it (~{count} photo(s) so far). Send more, or reply 'done' when finished.")
        return str(resp)

    resp.message("Send me photos (with optional captions), then reply 'done' when finished.")
    return str(resp)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


  init_db()

   if __name__ == "__main__":
       port = int(os.environ.get("PORT", 5000))
       app.run(host="0.0.0.0", port=port)
