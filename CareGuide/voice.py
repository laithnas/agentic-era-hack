from fastapi import APIRouter, UploadFile, HTTPException
from google.cloud import speech_v2, texttospeech
import base64

router = APIRouter(prefix="/voice", tags=["voice"])

def _stt_bytes_to_text(raw: bytes, lang="en-US"):
    client = speech_v2.SpeechClient()
    # assume LINEAR16 PCM 16kHz; change if you record WebM/OGG
    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=[lang],
        features=speech_v2.RecognitionFeatures(enable_automatic_punctuation=True),
        model="long",  # robust default
    )
    req = speech_v2.RecognizeRequest(
        recognizer=f"projects/{client.project}/locations/global/recognizers/_",
        config=config,
        content=raw,
    )
    resp = client.recognize(request=req)
    return resp.results[0].alternatives[0].transcript if resp.results else ""

def _tts_to_mp3(text: str, voice="en-US-Neural2-C"):
    tts = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice_sel = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice)
    audio_cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    audio = tts.synthesize_speech(input=synthesis_input, voice=voice_sel, audio_config=audio_cfg)
    return base64.b64encode(audio.audio_content).decode("ascii")

@router.post("/turn")
async def voice_turn(file: UploadFile):
    raw = await file.read()
    text = _stt_bytes_to_text(raw)
    if not text:
        raise HTTPException(400, "Couldn’t transcribe audio.")

    # Call your ADK agent via in-process import or HTTP invoke
    from .agent import root_agent
    events = []
    async for ev in root_agent.run_async({"messages":[{"role":"user","content":text}]}):
        events.append(ev)
    # Get the latest assistant text (simplified)
    reply = next((e.content for e in reversed(events) if getattr(e, "role", "") == "assistant" and getattr(e,"content",None)), "Sorry, I didn’t catch that.")
    mp3_b64 = _tts_to_mp3(reply)

    return {"asr_text": text, "reply_text": reply, "reply_mp3_base64": mp3_b64}
