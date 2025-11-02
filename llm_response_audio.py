from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("ayoubkirouane/whisper-small-ar")
model = AutoModelForSpeechSeq2Seq.from_pretrained("ayoubkirouane/whisper-small-ar")