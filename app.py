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

@app.post("/api/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    questions_per_set: int = Form(25)
):
    try:
        contents = await file.read()
        pdf_bytes = io.BytesIO(contents)
        
        all_extracted_text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    all_extracted_text += f"\n--- Page {i+1} ---\n" + page_text

        if not all_extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF se text read nahi ho saka.")

        prompt = f"""
        Task: Extract EVERY SINGLE multiple-choice question from ALL pages of the PDF text below.
        CRITICAL: Do NOT skip any question. Keep explanations short and concise to avoid output truncation. Do NOT use unescaped double quotes inside strings.

        Instructions:
        1. Extract all questions found across all pages.
        2. Provide Bilingual output (English and Hindi). Translate if missing.
        3. Provide 4 clear options (A, B, C, D), correct option, and short explanations.

        Return ONLY a valid JSON array of objects with these exact keys:
        - "id": integer
        - "question_en": string
        - "question_hi": string
        - "options_en": list of 4 strings (e.g. ["A) Option 1", "B) Option 2", ...])
        - "options_hi": list of 4 strings (e.g. ["A) विकल्प 1", "B) विकल्प 2", ...])
        - "correct_option": string ("A", "B", "C", or "D")
        - "explanation_en": string
        - "explanation_hi": string

        PDF TEXT TO PROCESS:
        {all_extracted_text}
        """

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=8192
            )
        )

        raw_text = response.text.strip()
        # Remove Markdown wrappers if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

        try:
            all_questions = json.loads(raw_text, strict=False)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500, 
                detail="PDF me questions zyada hain. 'Questions / Set' filter me kam number try karein ya PDF split karein."
            )

        if not all_questions:
            raise HTTPException(status_code=400, detail="PDF se questions parse nahi ho paye.")

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
    
