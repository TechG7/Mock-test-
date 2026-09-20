import os
import json
import io
import re
import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def parse_gemini_json(text):
    raw_text = text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)
    
    try:
        return json.loads(raw_text, strict=False)
    except Exception:
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0), strict=False)
        raise

def extract_text_from_file(filename: str, contents: bytes) -> list:
    pages_text = []
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            image = Image.open(io.BytesIO(contents))
            text = pytesseract.image_to_string(image, lang='eng+hin')
            if text.strip():
                pages_text.append(f"--- Image Content ---\n{text}")
        except Exception as e:
            print(f"Image OCR Error: {e}")
            
    elif ext == ".pdf":
        try:
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(f"--- Page {i+1} ---\n{text}")
        except Exception as e:
            print(f"PDF extraction error: {e}")

        # Fallback to OCR if PDF contains scanned images/no text
        if not pages_text:
            try:
                images = convert_from_bytes(contents)
                for i, image in enumerate(images):
                    text = pytesseract.image_to_string(image, lang='eng+hin')
                    if text.strip():
                        pages_text.append(f"--- Scanned Page {i+1} ---\n{text}")
            except Exception as ocr_err:
                print(f"PDF OCR Fallback Error: {ocr_err}")

    return pages_text

@app.post("/api/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    questions_per_set: int = Form(25),
    mode: str = Form("extract") # "extract" for MCQs paper, "generate" for Theory Chapter
):
    try:
        contents = await file.read()
        pages_text = extract_text_from_file(file.filename, contents)

        if not pages_text:
            raise HTTPException(status_code=400, detail="File se koi text ya image read nahi ho saka. File check karein.")

        BATCH_SIZE = 5
        all_questions = []
        global_id = 1

        for b in range(0, len(pages_text), BATCH_SIZE):
            batch_text = "\n\n".join(pages_text[b : b + BATCH_SIZE])
            
            if mode == "generate":
                prompt = f"""
                Task: Read the following Book Chapter/Theory content and CREATE high-quality multiple-choice questions (MCQs) testing important concepts.
                Instructions:
                1. Create as many relevant questions as possible from this text.
                2. Provide Bilingual output (English and Hindi).
                3. Provide 4 options (A, B, C, D), correct option, and clear explanation.
                4. Keep explanations short and concise.

                Return ONLY a valid JSON array of objects with keys:
                - "id": integer
                - "question_en": string
                - "question_hi": string
                - "options_en": list of 4 strings (e.g. ["A) ...", "B) ...", "C) ...", "D) ..."])
                - "options_hi": list of 4 strings (e.g. ["A) ...", "B) ...", "C) ...", "D) ..."])
                - "correct_option": string ("A", "B", "C", or "D")
                - "explanation_en": string
                - "explanation_hi": string

                BOOK TEXT:
                {batch_text}
                """
            else:
                prompt = f"""
                Task: Extract ALL multiple-choice questions from this document chunk.
                Instructions:
                1. Provide Bilingual output (English and Hindi). Translate if missing.
                2. Keep explanations short (max 1-2 lines).
                3. Do not use unescaped double quotes inside text strings.

                Return ONLY a valid JSON array of objects with keys:
                - "id": integer
                - "question_en": string
                - "question_hi": string
                - "options_en": list of 4 strings
                - "options_hi": list of 4 strings
                - "correct_option": string ("A", "B", "C", or "D")
                - "explanation_en": string
                - "explanation_hi": string

                DOCUMENT TEXT:
                {batch_text}
                """

            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=8192
                    )
                )
                
                chunk_qs = parse_gemini_json(response.text)
                if isinstance(chunk_qs, list):
                    for q in chunk_qs:
                        q["id"] = global_id
                        global_id += 1
                        all_questions.append(q)
            except Exception as batch_err:
                print(f"Batch {b} error: {batch_err}")
                continue

        if not all_questions:
            raise HTTPException(status_code=400, detail="Questions extract/generate nahi ho paaye.")

        sets = []
        total_questions = len(all_questions)
        
        for i in range(0, total_questions, questions_per_set):
            chunk = all_questions[i : i + questions_per_set]
            set_number = (i // questions_per_set) + 1
            sets.append({
                "set_id": set_number,
                "title": f"Set {set_number} (Q.{i+1} to Q.{i+len(chunk)})",
                "total_questions": len(chunk),
                "questions": chunk
            })

        return {
            "success": True,
            "total_questions_found": total_questions,
            "total_sets": len(sets),
            "sets": sets
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
