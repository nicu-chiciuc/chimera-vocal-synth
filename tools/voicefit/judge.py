"""Independent judges of fitted renders, never used by the fit itself (unless fit.py --speaker,
which uses the speaker judge on the training phrases only):

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
import torch
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compare  # noqa: E402

CACHE = pathlib.Path.home() / ".cache" / "chimera-target" / "models"
NORM = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])


def spans(track, which):
    for i, p in enumerate(track["phrases"]):
        if which is None or i in which:
            ph = [x for w in p["words"] for x in w["phones"]]
            yield p["text"], ph[0][1] - 0.1, ph[-1][2] + 0.2


class Judge:
    def __init__(self, d):
        self.d = pathlib.Path(d)
        self.track = json.load(open(self.d / "track.json"))
        self.enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=str(CACHE / "ecapa"), run_opts={"device": "cpu"})
        self.asr = WhisperModel("small.en", device="cpu", compute_type="int8")
        self.target = librosa.load(self.d / "target.wav", sr=16000)[0]
        self._ref = {}

    def _embed(self, y, parts):
        with torch.no_grad():
            e = torch.stack([self.enc.encode_batch(torch.tensor(y[int(a * 16000) : int(b * 16000)])[None]).squeeze() for _, a, b in parts]).mean(0)
        return e / e.norm()

    def judge(self, wav, which):
        """{similarity, wer, heard} of a render over the chosen phrases."""
        parts = list(spans(self.track, which))
        key = tuple(sorted(which)) if which else None
        if key not in self._ref:
            self._ref[key] = self._embed(self.target, parts)
        y = librosa.load(wav, sr=16000)[0]
        sim = float(self._ref[key] @ self._embed(y, parts))
        heard = []
        for _, a, b in parts:
            segs, _ = self.asr.transcribe(y[int(a * 16000) : int(b * 16000)], language="en", beam_size=5)
            heard.append(" ".join(s.text for s in segs).strip())
        ref = " ".join(t for t, _, _ in parts)
        wer = jiwer.wer(ref, " ".join(heard), reference_transform=NORM, hypothesis_transform=NORM)
        return {"similarity": round(sim, 4), "wer": round(wer, 4), "heard": heard}


def main():
    args = sys.argv[1:]
    which = None
    if "--phrases" in args:
        i = args.index("--phrases")
        which = compare.parse_which(args[i + 1])
        del args[i : i + 2]
    j = Judge(args[0])
    for wav in [str(j.d / "target.wav")] + args[1:]:
        r = j.judge(wav, which)
        print(f"{pathlib.Path(wav).name:22s} speaker similarity {r['similarity']:5.3f}   word error {100 * r['wer']:5.1f} %   heard: {' | '.join(r['heard'])}")


if __name__ == "__main__":
    main()
