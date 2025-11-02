# voice_two_endpoints.py
# Fixed version - Voice Command Service with RAG

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

# Optional HF TTS + dependencies
try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    import torch
    import soundfile as sf
    HF_TTS_AVAILABLE = True
except Exception:
    HF_TTS_AVAILABLE = False

# gTTS fallback
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
VOSK_MODEL_PATH = "vosk_model/vosk-model/vosk-model"
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
-"/dashboard" : This screen contains data about soil moisture, temperature, humidity and saved water. It also contains you recent activity.
- "/soil" : This screen contains soil parameters : Moisture level , ph level , Temperature and conductivity.
- "/weather": This screen contains weather forecast for the next 7 days.
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
# TTS: HF attempt then gTTS fallback
# ----------------------------
TTS_MODEL_NAME = "MBZUAI/speecht5_tts_clartts_ar"
TTS_SAMPLE_RATE = 24000

_tts_processor = None
_tts_model = None
_tts_device = None

def _load_hf_tts(model_name: str = TTS_MODEL_NAME):
    global _tts_processor, _tts_model, _tts_device
    if _tts_processor is None:
        if not HF_TTS_AVAILABLE:
            raise RuntimeError("HF TTS deps not installed")
        _tts_processor = AutoProcessor.from_pretrained(model_name)
        _tts_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        _tts_device = "cuda" if torch.cuda.is_available() else "cpu"
        _tts_model.to(_tts_device)

def _float_array_to_wav_bytes(arr, sample_rate=TTS_SAMPLE_RATE):
    import io
    import numpy as np
    if not HF_TTS_AVAILABLE:
        raise RuntimeError("soundfile not available")
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=0)
    buf = io.BytesIO()
    sf.write(buf, a, samplerate=sample_rate, subtype='PCM_16', format='WAV')
    buf.seek(0)
    return buf.read()

def synthesize_tts_hf(text: str) -> Optional[bytes]:
    try:
        _load_hf_tts()
        inputs = _tts_processor(text, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(_tts_device)
        with torch.no_grad():
            generated = _tts_model.generate(**inputs, max_length=10000)
        
        try:
            audio_arr = _tts_processor.decode(generated[0].cpu().numpy())
            return _float_array_to_wav_bytes(audio_arr, sample_rate=TTS_SAMPLE_RATE)
        except Exception:
            arr = generated[0].cpu().numpy()
            return _float_array_to_wav_bytes(arr, sample_rate=TTS_SAMPLE_RATE)
    except Exception as e:
        print(f"❌ HF TTS error: {e}")
        return None

def synthesize_tts_gtts(text: str) -> Optional[bytes]:
    if not GTTS_AVAILABLE or not PYDUB_AVAILABLE:
        return None
    
    tmp_mp3_path = None
    tmp_wav_path = None
    try:
        tts = gTTS(text=text, lang="ar", slow=False)
        tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_mp3_path = tmp_mp3.name
        tmp_mp3.close()
        
        tts.save(tmp_mp3_path)
        audio = AudioSegment.from_file(tmp_mp3_path, format="mp3")
        audio = audio.set_frame_rate(TTS_SAMPLE_RATE).set_channels(1).set_sample_width(2)
        
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()
        
        audio.export(tmp_wav_path, format="wav")
        with open(tmp_wav_path, "rb") as f:
            wav_bytes = f.read()
        
        return wav_bytes
    except Exception as e:
        print(f"❌ gTTS error: {e}")
        return None
    finally:
        # Clean up temp files
        for path in [tmp_mp3_path, tmp_wav_path]:
            if path:
                try:
                    os.unlink(path)
                except:
                    pass

def synthesize_tts(text: str) -> Optional[bytes]:
    """Try HF TTS first, fall back to gTTS"""
    if HF_TTS_AVAILABLE:
        wav = synthesize_tts_hf(text)
        if wav:
            return wav
    
    if GTTS_AVAILABLE and PYDUB_AVAILABLE:
        wav = synthesize_tts_gtts(text)
        if wav:
            return wav
    
    return None

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
# ENDPOINT 2: /rag_tts (Context + English Query -> Response + Audio)
# ----------------------------
@app.post("/rag_tts")
async def rag_tts_endpoint(body: RagTtsRequest):
    """
    Accept context and English query, generate response and TTS audio.
    Query should be the english_text from /voice endpoint.
    Returns: {response_text, audio_base64, sample_rate}
    """
    if not body.query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    
    try:
        print(f"📝 RAG query (English): {body.query}")
        
        # Generate RAG text using English query
        rag = generate_rag_text(body.context, body.query)
        response_text = rag["response_text"]
        
        print(f"💬 Response text (Arabic): {response_text}")

        # Synthesize TTS
        wav_bytes = synthesize_tts(response_text)
        audio_b64 = None
        if wav_bytes:
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            print(f"🔊 TTS audio generated")

        return JSONResponse(content={
            "response_text": response_text,
            "audio_base64": audio_b64,
            "sample_rate": TTS_SAMPLE_RATE if wav_bytes else None
        })
    except Exception as e:
        print(f"❌ RAG TTS error: {e}")
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
            "/rag_tts": "POST - Get RAG response with TTS audio (use english_text from /voice)"
        },
        "dependencies": {
            "pydub": PYDUB_AVAILABLE,
            "gtts": GTTS_AVAILABLE,
            "hf_tts": HF_TTS_AVAILABLE
        }
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
    print("  POST /rag_tts - English query + context -> Arabic response + TTS")
    print("\nDependencies:")
    print(f"  pydub: {PYDUB_AVAILABLE}")
    print(f"  gTTS: {GTTS_AVAILABLE}")
    print(f"  HF TTS: {HF_TTS_AVAILABLE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)