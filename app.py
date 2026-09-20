import os
import json
import pdfplumber
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Client Setup (Set your GEMINI_API_KEY environment variable)
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
        # 1. Read PDF Text
        pdf_text = extract_text_from_pdf(file.file)
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="Could not read text from PDF.")

        # 2. Prompt Gemini API to parse MCQs into JSON
        prompt = f"""
        Extract all multiple choice questions from the following text and return ONLY a JSON array.
        Each object must have:
        - "id": integer
        - "question": string
        - "options": list of 4 strings (e.g. ["A) Option 1", "B) Option 2", ...])
        - "correct_option": string (e.g. "A", "B", "C", or "D")
        - "explanation": string (brief explanation of the answer)

        PDF Content:
        {pdf_text[:15000]}  # Limiting token length for stability
        """

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        all_questions = json.loads(response.text)

        # 3. Split questions into Sets (Chunking)
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
    
