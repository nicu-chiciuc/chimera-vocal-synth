"""Independent judges of fitted renders, never used by the fit itself:

  judge.py <dir> <render.wav>... [--phrases 4-7]

- speaker similarity: cosine between ECAPA speaker embeddings (SpeechBrain spkrec-ecapa-voxceleb)
  of the singer's phrases and the render's same phrases. Same speaker in VoxCeleb is roughly
  above 0.25; the number is for comparing renders with each other, not a verdict on identity.
- intelligibility: Whisper (small.en) word error rate on the render's phrases against the words
  the singer sang (the track's text).
"""
import json
import pathlib
import sys

import jiwer
import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compare  # noqa: E402

CACHE = pathlib.Path.home() / ".cache" / "chimera-target" / "models"


def spans(track, which):
    for i, p in enumerate(track["phrases"]):
        if which is None or i in which:
            ph = [x for w in p["words"] for x in w["phones"]]
            yield p["text"], ph[0][1] - 0.1, ph[-1][2] + 0.2


def main():
    args = sys.argv[1:]
    which = None
    if "--phrases" in args:
        i = args.index("--phrases")
        which = compare.parse_which(args[i + 1])
        del args[i : i + 2]
    d = pathlib.Path(args[0])
    track = json.load(open(d / "track.json"))
    parts = list(spans(track, which))
    enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=str(CACHE / "ecapa"), run_opts={"device": "cpu"})
    asr = WhisperModel("small.en", device="cpu", compute_type="int8")

    def embed(y):
        e = [enc.encode_batch(torch.tensor(y[int(a * 16000) : int(b * 16000)])[None]).squeeze() for _, a, b in parts]
        e = torch.stack(e).mean(0)
        return e / e.norm()

    target = librosa.load(d / "target.wav", sr=16000)[0]
    et = embed(target)
    ref = " ".join(t for t, _, _ in parts)
    rows = [("target", target)] + [(pathlib.Path(w).name, librosa.load(w, sr=16000)[0]) for w in args[1:]]
    for name, y in rows:
        sim = float(et @ embed(y))
        hyp = []
        for _, a, b in parts:
            segs, _ = asr.transcribe(y[int(a * 16000) : int(b * 16000)], language="en", beam_size=5)
            hyp.append(" ".join(s.text for s in segs))
        norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
        wer = jiwer.wer(ref, " ".join(hyp), reference_transform=norm, hypothesis_transform=norm)
        print(f"{name:22s} speaker similarity {sim:5.3f}   word error {100 * wer:5.1f} %   heard: {' | '.join(h.strip() for h in hyp)}")


if __name__ == "__main__":
    main()
