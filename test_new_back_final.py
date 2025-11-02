# voice_two_endpoints.py
# Fixed version - Voice Command Service with Working TTS (MP3 output)

import os
import json
import base64
import tempfile
import wave
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# LLM client (OpenRouter-compatible)
from openai import OpenAI

# ASR (Vosk)
from vosk import Model, KaldiRecognizer

# audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    AudioSegment = None
    PYDUB_AVAILABLE = False

# gTTS for Arabic TTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# ----------------------------
# CONFIG
# ----------------------------

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

# Vosk model path
VOSK_MODEL_PATH = "vosk_model/vosk-model"
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")
asr_model = Model(VOSK_MODEL_PATH)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Voice Command Service")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"]
)

# ----------------------------
# Pydantic request models
# ----------------------------
class RagTtsRequest(BaseModel):
    context: Dict[str, Any] = {}
    query: str

# ----------------------------
# Few-shot examples & system prompts
# ----------------------------
FEW_SHOT_EXAMPLES = [
    {
        "user": "nheb nchouf el ta9s el youm", 
        "assistant": {
            "valid": True,
            "corrected_text": "نحب نشوف الطقس اليوم",
            "english_text": "I want to see today's weather",
            "intent": "view_weather",
            "route": "/weather"
        }
    },
    {
        "user": "ajout tache jdida lel khetra", 
        "assistant": {
            "valid": True,
            "corrected_text": "أضيف مهمة جديدة للحقول",
            "english_text": "Add a new task for the fields",
            "intent": "add_task",
            "route": "/tasks/new"
        }
    },
    {
        "user": "chbih Messi elyom ?", 
        "assistant": {
            "valid": False,
            "message": "Please repeat, I didn't understand."
        }
    }
]

ROUTE_SYSTEM_PROMPT = """
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

RAG_SYSTEM_PROMPT = """
You are a knowledgeable agriculture assistant.
You will be given:
- CONTEXT: JSON app UI data (fields, weather, tasks, mappings)
- USER QUERY: user text in English (already translated from Tunisian Arabic)

Produce strict JSON response:
{
  "response_text": "..."
}
- response_text: final conversational answer in Arabic suitable for TTS
If context lacks info, ask follow-up or say you need more info.
"""

# ----------------------------
# Helper: Transcribe Audio
# ----------------------------
def transcribe_audio(file_path: str) -> str:
    """Transcribe audio file using Vosk ASR"""
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be WAV format mono PCM")
    rec = KaldiRecognizer(asr_model, wf.getframerate())
    rec.AcceptWaveform(wf.readframes(wf.getnframes()))
    text = json.loads(rec.FinalResult())["text"]
    return text

# ----------------------------
# Helper: LLM Route Mapping
# ----------------------------
def process_text_with_llm(raw_text: str) -> Dict[str, Any]:
    """Process transcribed text and map to route"""
    few_shot_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_str += f"User: {ex['user']}\nAssistant: {json.dumps(ex['assistant'], ensure_ascii=False)}\n\n"

    prompt = f"{ROUTE_SYSTEM_PROMPT}\n\n{few_shot_str}User: {raw_text}\nAssistant:"

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    llm_output_text = response.choices[0].message.content.strip()
    print(f"🤖 LLM raw output: {llm_output_text}")
    
    try:
        return json.loads(llm_output_text)
    except json.JSONDecodeError:
        return {"valid": False, "message": "LLM output could not be parsed"}

# ----------------------------
# Helper: Generate RAG Response
# ----------------------------
def generate_rag_text(context: Dict[str, Any], query: str, model_name: str = "deepseek/deepseek-chat-v3.1") -> Dict[str, str]:
    """Generate contextual response using RAG"""
    formatted_context = json.dumps(context, ensure_ascii=False, indent=2)
    prompt = f"{RAG_SYSTEM_PROMPT}\n\nCONTEXT:\n{formatted_context}\n\nUSER QUERY (English):\n{query}\n\nRespond now in JSON format."
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        out = resp.choices[0].message.content.strip()
        print(f"🤖 RAG raw output: {out}")
        
        # Parse response
        try:
            parsed = json.loads(out)
            response_text = parsed.get("response_text", "").strip()
            if response_text:
                return {"response_text": response_text}
        except json.JSONDecodeError:
            pass
        
        # Fallback: use raw output if no JSON
        return {"response_text": out}
    except Exception as e:
        print(f"❌ RAG error: {e}")
        return {"response_text": "عذراً، حدث خطأ في معالجة طلبك."}

# ----------------------------
# TTS: Direct MP3 output with gTTS
# ----------------------------
def synthesize_tts_mp3(text: str) -> Optional[bytes]:
    """
    Generate Arabic TTS audio using gTTS and return MP3 bytes directly.
    No conversion needed - gTTS produces MP3 natively.
    """
    if not GTTS_AVAILABLE:
        print("❌ gTTS not available")
        return None
    
    tmp_mp3_path = None
    try:
        print(f"🎤 Generating TTS for: {text[:50]}...")
        
        # Create gTTS object with Arabic language
        tts = gTTS(text=text, lang="ar", slow=False)
        
        # Save to temporary MP3 file
        tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_mp3_path = tmp_mp3.name
        tmp_mp3.close()
        
        tts.save(tmp_mp3_path)
        print(f"💾 TTS saved to: {tmp_mp3_path}")
        
        # Read MP3 bytes
        with open(tmp_mp3_path, "rb") as f:
            mp3_bytes = f.read()
        
        print(f"✅ TTS generated: {len(mp3_bytes)} bytes")
        return mp3_bytes
        
    except Exception as e:
        print(f"❌ gTTS error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temp file
        if tmp_mp3_path and os.path.exists(tmp_mp3_path):
            try:
                os.unlink(tmp_mp3_path)
                print(f"🗑️ Cleaned up: {tmp_mp3_path}")
            except Exception as e:
                print(f"⚠️ Could not delete temp file: {e}")

# ----------------------------
# ENDPOINT 1: /voice (Audio -> ASR -> Route Mapping)
# ----------------------------
@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    """
    Accept audio file, transcribe it, then map to route.
    Returns: {transcribed_text, valid, corrected_text, english_text, intent, route}
    """
    temp_input = None
    temp_wav = None
    try:
        print(f"📥 Received file: {file.filename}")
        
        # Save uploaded file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
        content = await file.read()
        print(f"📦 File size: {len(content)} bytes")
        temp_input.write(content)
        temp_input.close()
        
        print(f"💾 Saved temp file: {temp_input.name}")
        
        # Convert to WAV (mono, 16kHz, 16-bit PCM)
        try:
            audio = AudioSegment.from_file(temp_input.name)
            print(f"🎵 Original audio: {audio.channels}ch, {audio.frame_rate}Hz, {len(audio)}ms")
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio.export(temp_wav.name, format='wav')
            temp_wav.close()
            
            print(f"✅ Converted to WAV: {temp_wav.name}")
        except Exception as e:
            print(f"❌ Audio conversion failed: {str(e)}")
            raise ValueError(f"Could not convert audio file: {str(e)}")

        # 1️⃣ Transcribe
        try:
            raw_text = transcribe_audio(temp_wav.name)
            print(f"📝 Transcribed text: '{raw_text}'")

            if not raw_text:
               print("⚠️ No speech detected in audio")
               return JSONResponse(content={
                  "transcribed_text": "",
                  "valid": False,
                  "message": "No speech detected"
               })
        except Exception as e:
            print(f"❌ Transcription failed: {str(e)}")
            raise ValueError(f"Transcription error: {str(e)}")

        # 2️⃣ Process with LLM (route mapping)
        try:
            result = process_text_with_llm(raw_text)
            result["transcribed_text"] = raw_text
            print(f"🎯 Final result: {result}")
            
            return JSONResponse(content=result)
        except Exception as e:
            print(f"❌ LLM processing failed: {str(e)}")
            raise ValueError(f"LLM error: {str(e)}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(content={"valid": False, "message": str(e)})
    
    finally:
        # Clean up temp files
        for temp_file in [temp_input, temp_wav]:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                    print(f"🗑️ Cleaned up: {temp_file.name}")
                except:
                    pass

# ----------------------------
# ENDPOINT 2: /rag_tts (Context + English Query -> Response + MP3 Audio)
# ----------------------------
@app.post("/rag_tts")
async def rag_tts_endpoint(body: RagTtsRequest):
    """
    Accept context and English query, generate response and TTS audio.
    Query should be the english_text from /voice endpoint.
    Returns: {response_text, audio_base64, audio_format}
    """
    if not body.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    
    try:
        print(f"📝 RAG query (English): {body.query}")
        
        # Generate RAG text using English query
        rag = generate_rag_text(body.context, body.query)
        response_text = rag["response_text"]
        
        print(f"💬 Response text (Arabic): {response_text}")

        # Synthesize TTS to MP3
        mp3_bytes = synthesize_tts_mp3(response_text)
        audio_b64 = None
        
        if mp3_bytes:
            audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
            print(f"🔊 TTS MP3 audio generated: {len(audio_b64)} chars (base64)")
        else:
            print("⚠️ TTS generation failed - no audio produced")

        return JSONResponse(content={
            "response_text": response_text,
            "audio_base64": audio_b64,
            "audio_format": "mp3" if mp3_bytes else None
        })
    except Exception as e:
        print(f"❌ RAG TTS error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Health check endpoint
# ----------------------------
@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Voice Command Service is active",
        "endpoints": {
            "/voice": "POST - Upload audio file for transcription and routing",
            "/rag_tts": "POST - Get RAG response with MP3 TTS audio (use english_text from /voice)"
        },
        "dependencies": {
            "pydub": PYDUB_AVAILABLE,
            "gtts": GTTS_AVAILABLE
        },
        "audio_format": "MP3"
    }

# ----------------------------
# Run local server
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting Voice Command Service...")
    print("📡 Server will be available at: http://0.0.0.0:8000")
    print("📚 API docs available at: http://0.0.0.0:8000/docs")
    print("\nEndpoints:")
    print("  POST /voice - Audio file -> transcription -> routing")
    print("  POST /rag_tts - English query + context -> Arabic response + MP3 TTS")
    print("\nDependencies:")
    print(f"  pydub: {PYDUB_AVAILABLE}")
    print(f"  gTTS: {GTTS_AVAILABLE}")
    print("\n⚠️  Make sure gTTS is installed: pip install gtts")
    uvicorn.run(app, host="0.0.0.0", port=8000)