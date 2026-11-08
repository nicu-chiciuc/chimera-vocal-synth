"""Word timings of a vocal with faster-whisper: transcribe.py <vocal.wav> <out.json>"""
import json
import sys

import librosa
from faster_whisper import WhisperModel

audio, _ = librosa.load(sys.argv[1], sr=16000, mono=True)
model = WhisperModel("small.en", device="cpu", compute_type="int8")
segs, _ = model.transcribe(audio, word_timestamps=True, vad_filter=True, language="en")
words = [{"w": w.word.strip(), "t0": round(w.start, 3), "t1": round(w.end, 3)} for s in segs for w in s.words]
json.dump(words, open(sys.argv[2], "w"), indent=0)
for w in words:
    print(f"{w['t0']:7.2f} {w['t1']:7.2f} {w['w']}")
