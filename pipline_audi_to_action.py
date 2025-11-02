# ====================================================
# Local Voice-to-Route Service (Vosk + LLM + FastAPI)
# ====================================================

import os
import wave
import json
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
from openai import OpenAI
import uvicorn
from pydub import AudioSegment

# -------------------------
# 1️⃣ Set OpenRouter API Key
# -------------------------
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

# -------------------------
# 2️⃣ Load Vosk ASR Model
# -------------------------
vosk_model_path = "vosk_model/vosk-model"  # adjust to your local folder
if not os.path.exists(vosk_model_path):
    raise FileNotFoundError("Vosk model folder not found at " + vosk_model_path)
model = Model(vosk_model_path)

# -------------------------
# 3️⃣ Few-shot examples & system prompt
# -------------------------
few_shot_examples = [
    {"user": "nheb nchouf el ta9s el youm", 
     "assistant": {
         "valid": True,
         "corrected_text": "نحب نشوف الطقس اليوم",
         "english_text": "I want to see today's weather",
         "intent": "view_weather",
         "route": "/weather"
     }},
    {"user": "ajout tache jdida lel khetra", 
     "assistant": {
         "valid": True,
         "corrected_text": "أضيف مهمة جديدة للحقول",
         "english_text": "Add a new task for the fields",
         "intent": "add_task",
         "route": "/tasks/new"
     }},
    {"user": "chbih Messi elyom ?", 
     "assistant": {
         "valid": False,
         "message": "Please repeat, I didn't understand."
     }}
]

system_prompt = """
You are an assistant for an agriculture mobile app. Your job is to:
1. Correct spelling and dialect of user input.
2. Translate it to English.
3. Extract the intent (what the user wants to do in the app) and map it to a Flutter route.
4. If the text is NOT related to the app/agriculture, ask the user to repeat.

Output must always be JSON.

If related:
{
  "valid": true,
  "corrected_text": "...",
  "english_text": "...",
  "intent": "...",
  "route": "..."
}

If unrelated:
{
  "valid": false,
  "message": "Please repeat, I didn't understand."
}

Available intents and routes:
- view_crops → "/crops"
- view_weather → "/weather"
- view_calendar → "/calendar"
- add_task → "/tasks/new"
- go_home → "/home"
"""

# -------------------------
# 4️⃣ FastAPI App
# -------------------------
app = FastAPI(title="Voice Command Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 5️⃣ Helper: Transcribe Audio
# -------------------------
def transcribe_audio(file_path):
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be WAV format mono PCM")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.AcceptWaveform(wf.readframes(wf.getnframes()))
    text = json.loads(rec.FinalResult())["text"]
    return text

# -------------------------
# 6️⃣ Helper: LLM Processing
# -------------------------
def process_text_with_llm(raw_text):
    few_shot_str = ""
    for ex in few_shot_examples:
        few_shot_str += f"User: {ex['user']}\nAssistant: {json.dumps(ex['assistant'], ensure_ascii=False)}\n\n"

    prompt = f"{system_prompt}\n\n{few_shot_str}\nUser: {raw_text}\nAssistant:"

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    llm_output_text = response.choices[0].message.content.strip()
    try:
        return json.loads(llm_output_text)
    except json.JSONDecodeError:
        return {"valid": False, "message": "LLM output could not be parsed"}

# -------------------------
# 7️⃣ Endpoint: /voice
# -------------------------
# -------------------------
# 7️⃣ Endpoint: /voice
# -------------------------
@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    temp_input = None
    temp_wav = None
    try:
        # Save uploaded file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
        temp_input.write(await file.read())
        temp_input.close()
        
        # Convert to WAV (mono, 16kHz, 16-bit PCM)
        audio = AudioSegment.from_file(temp_input.name)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.export(temp_wav.name, format='wav')
        temp_wav.close()
        
        print(f"✅ Converted to WAV: {temp_wav.name}")

        # 1️⃣ Transcribe
        raw_text = transcribe_audio(temp_wav.name)
        print(f"📝 Transcribed text: {raw_text}")

        # 2️⃣ Process with LLM
        result = process_text_with_llm(raw_text)
        print(f"🤖 LLM result: {result}")

        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(content={"valid": False, "message": str(e)})
    
    finally:
        # Clean up temp files
        for temp_file in [temp_input, temp_wav]:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

# -------------------------
# 8️⃣ Health check endpoint
# -------------------------
@app.get("/")
async def root():
    return {"status": "running", "message": "Voice Command Service is active"}

# -------------------------
# 9️⃣ Run local server
# -------------------------
if __name__ == "__main__":
    print("🚀 Starting Voice Command Service...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)