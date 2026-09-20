import os
import json
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

def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

@app.post("/api/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    questions_per_set: int = Form(25)
):
    try:
        pdf_text = extract_text_from_pdf(file.file)
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="PDF में टेक्स्ट नहीं मिला।")

        prompt = f"""
        Extract all multiple-choice questions from the provided text and convert them into a Bilingual (English and Hindi) JSON array.
        If the original text is only in English or Hindi, translate and provide both versions.

        Return ONLY a JSON array where each object has:
        - "id": integer
        - "question_en": string (English)
        - "question_hi": string (Hindi)
        - "options_en": list of 4 strings (e.g. ["A) Option 1", "B) Option 2", ...])
        - "options_hi": list of 4 strings (e.g. ["A) विकल्प 1", "B) विकल्प 2", ...])
        - "correct_option": string (e.g. "A", "B", "C", or "D")
        - "explanation_en": string (brief explanation in English)
        - "explanation_hi": string (brief explanation in Hindi)

        PDF Content:
        {pdf_text[:20000]}
        """

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        all_questions = json.loads(response.text)

        sets = []
        total_questions = len(all_questions)
        
        for i in range(0, total_questions, questions_per_set):
            chunk = all_questions[i : i + questions_per_set]
            set_number = (i // questions_per_set) + 1
            sets.append({
                "set_id": set_number,
                "title": f"Set {set_number} (Q.{i+1} to Q.{i+len(chunk)})",
                "total_questions": len(chunk),
                "questions": chunk,
                "status": "pending"
            })

        return {
            "success": True,
            "total_questions_found": total_questions,
            "total_sets": len(sets),
            "sets": sets
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
