import os
import json
import io
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
        # 1. Read full PDF binary data using io.BytesIO
        contents = await file.read()
        pdf_bytes = io.BytesIO(contents)
        
        all_extracted_text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    all_extracted_text += f"\n--- Page {i+1} ---\n" + page_text

        if not all_extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF se text read nahi ho saka. Text/scanned PDF check karein.")

        # 2. Strict Prompt forcing AI to parse EVERY single question across all pages
        prompt = f"""
        Task: You are an expert exam generator. Extract EVERY SINGLE multiple-choice question from ALL pages of the PDF text below.
        CRITICAL: Do NOT skip any question. Do NOT stop early. Read from Page 1 to the last page.

        Instructions:
        1. Extract all questions found across all pages.
        2. Provide Bilingual output (English and Hindi). Translate if the original text is in only one language.
        3. Make sure every question has 4 clear options (A, B, C, D) and a correct answer key with explanation.

        Return ONLY a valid JSON array of objects with these exact keys:
        - "id": integer (1, 2, 3...)
        - "question_en": string
        - "question_hi": string
        - "options_en": list of 4 strings (e.g. ["A) ...", "B) ...", "C) ...", "D) ..."])
        - "options_hi": list of 4 strings (e.g. ["A) ...", "B) ...", "C) ...", "D) ..."])
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

        all_questions = json.loads(response.text)

        if not all_questions:
            raise HTTPException(status_code=400, detail="PDF me se koi questions extract nahi ho paye.")

        # 3. Auto Chunking into Multiple Sets
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
    
