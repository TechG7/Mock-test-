import os
import json
import io
import re
import pdfplumber
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

@app.post("/api/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    questions_per_set: int = Form(25)
):
    try:
        contents = await file.read()
        pdf_bytes = io.BytesIO(contents)
        
        pages_text = []
        with pdfplumber.open(pdf_bytes) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(f"--- Page {i+1} ---\n{text}")

        if not pages_text:
            raise HTTPException(status_code=400, detail="PDF se text read nahi ho saka.")

        # Batch processing: 5 pages per API call to avoid token truncation
        BATCH_SIZE = 5
        all_questions = []
        global_id = 1

        for b in range(0, len(pages_text), BATCH_SIZE):
            batch_text = "\n\n".join(pages_text[b : b + BATCH_SIZE])
            
            prompt = f"""
            Task: Extract ALL multiple-choice questions from this PDF chunk.
            Instructions:
            1. Provide Bilingual output (English and Hindi). Translate if missing.
            2. Keep explanations short (max 1-2 lines).
            3. Do not use unescaped double quotes inside text strings.

            Return ONLY a valid JSON array of objects with keys:
            - "id": integer
            - "question_en": string
            - "question_hi": string
            - "options_en": list of 4 strings (e.g. ["A) ...", "B) ...", "C) ...", "D) ..."])
            - "options_hi": list of 4 strings (e.g. ["A) ...", "B) ...", "C) ...", "D) ..."])
            - "correct_option": string ("A", "B", "C", or "D")
            - "explanation_en": string
            - "explanation_hi": string

            PDF CHUNK TEXT:
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
            raise HTTPException(status_code=400, detail="PDF se questions extract nahi ho paaye. PDF quality check karein.")

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
    
