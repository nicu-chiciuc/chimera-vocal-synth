"""Time the words of a sung vocal with a CTC aligner, then hand MFA one word at a time:

  align.py <dir>

<dir> holds target.wav and lines.json (prepare.py cut). The Montreal Forced Aligner is a speech
model; on a singer who holds notes over a band's harmonies it puts the sustain on the wrong word
(Country Roads: "to the place I belong" came out with 2.45 s of "the" and 0.23 s of "belong",
and the held "roads" of "country roads, take me home" was taken for 2 s of silence). A CTC
acoustic model (wav2vec2, what lyric-alignment tools use) tolerates sustains: it emits each
letter once near its onset and blanks through the rest.

So: every word of the lyrics is force-aligned against the whole excerpt with wav2vec2's CTC
output, which gives each word's onset (its first letter's emission). A word ends where the
next one starts, or earlier at a pause (the level under the vowel floor for 150 ms or more).
The MFA corpus is then rewritten as one utterance per word, each cut to its span (plus a small
margin), so MFA only places the phones inside a word whose time is already known. prepare.py
collect reads that layout from lines.json ("words" on each line).

Prints the words whose CTC span differs most from MFA's earlier line-level guess, if there was
one, and writes dir/words-ctc.json (the spans) for inspection.
"""
import json
import pathlib
import re
import shutil
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.functional import forced_align, merge_tokens

SR = 16000
HOP = 0.01  # the level envelope's step
PAUSE = 0.15  # a dip this long between two words ends the first one there
FLOOR_DB = 25.0  # a pause is this far under the median level of the words' onsets
DROP_DB = 18.0  # or this far under the word's own peak
PAD_BEFORE = 0.10  # margin around each word's audio for MFA (CTC onsets run a little late)
PAD_AFTER = 0.05
MIN_UTT = 0.25  # MFA gets at least this much audio per word

norm = lambda w: re.sub(r"[^A-Z']", "", w.upper().replace("’", "'").replace("‘", "'"))  # noqa: E731


def ctc_word_onsets(y, words):
    """Onset time of each word (first letter's emission) and its last letter's emission time."""
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
    labels = bundle.get_labels()
    index = {c: i for i, c in enumerate(labels)}
    wave = torch.tensor(y, dtype=torch.float32)[None]
    with torch.inference_mode():
        emissions, _ = model(wave)
        emissions = torch.log_softmax(emissions, dim=-1)
    ratio = wave.shape[1] / emissions.shape[1] / SR  # seconds per emission frame
    transcript = "|".join(norm(w) for w in words)
    tokens = [index[c] for c in transcript]
    aligned, scores = forced_align(emissions, torch.tensor([tokens], dtype=torch.int32), blank=0)
    spans = merge_tokens(aligned[0], scores[0].exp())
    # spans follow the transcript's characters in order; group them into words at the '|'
    out, cur = [], []
    for s in spans:
        if labels[s.token] == "|":
            out.append(cur)
            cur = []
        else:
            cur.append(s)
    out.append(cur)
    assert len(out) == len(words), (len(out), len(words))
    return [(sp[0].start * ratio, (sp[-1].end + 1) * ratio) for sp in out]


def main(d):
    d = pathlib.Path(d)
    meta = json.load(open(d / "lines.json"))
    y, _ = librosa.load(d / "target.wav", sr=SR, mono=True)
    words = [(k, w) for k, line in enumerate(meta["lines"]) for w in line["text"].split()]
    spans = ctc_word_onsets(y, [w for _, w in words])
    onsets = [a for a, _ in spans]

    # level, 10 ms: a pause is a run under the floor between one word's last letter and the next onset
    rms = librosa.feature.rms(y=y, frame_length=int(0.025 * SR), hop_length=int(HOP * SR))[0]
    db = 20 * np.log10(rms + 1e-9)
    at = lambda t: db[min(len(db) - 1, int(t / HOP))]  # noqa: E731
    floor = np.median([at(a + 0.05) for a in onsets]) - FLOOR_DB
    ends = []
    for i, (a, b) in enumerate(spans):
        nxt = onsets[i + 1] if i + 1 < len(spans) else min(len(y) / SR, b + 1.5)
        end = nxt
        i0, i1 = int(b / HOP), int(nxt / HOP)
        # a pause: under the floor, or well under the word's own peak (a tail of reverb and the
        # band's leak into the vocal stem stays above the floor)
        peak = db[int(a / HOP) : max(i0 + 1, int(a / HOP) + 1)].max()
        quiet = db[i0:i1] < max(floor, peak - DROP_DB)
        run = 0
        for k, q in enumerate(quiet):
            run = run + 1 if q else 0
            if run * HOP >= PAUSE:
                end = (i0 + k - run + 1) * HOP
                break
        ends.append(max(end, b, a + 0.05))

    # the MFA corpus: one utterance per word
    corpus = d / "mfa" / "corpus"
    for old in (corpus, d / "mfa" / "aligned"):
        if old.exists():
            shutil.rmtree(old)
    corpus.mkdir(parents=True)
    per_line: dict[int, list] = {}
    dur = len(y) / SR
    for j, ((k, w), a, e) in enumerate(zip(words, onsets, ends)):
        t0 = max(0.0, a - PAD_BEFORE)
        t1 = min(dur, max(e + PAD_AFTER, t0 + MIN_UTT))
        name = f"line{k + 1:02d}w{len(per_line.get(k, [])) + 1:02d}"
        u = corpus / name
        u.mkdir()
        sf.write(u / f"{name}.wav", y[int(t0 * SR) : int(t1 * SR)], SR)
        (u / f"{name}.lab").write_text(w + "\n")
        per_line.setdefault(k, []).append({"name": name, "w": w, "t0": round(t0, 3), "t1": round(t1, 3), "on": round(a, 3), "off": round(e, 3)})
    for k, line in enumerate(meta["lines"]):
        line["words"] = per_line.get(k, [])
        if line["words"]:
            line["t0"], line["t1"] = line["words"][0]["t0"], line["words"][-1]["t1"]
    json.dump(meta, open(d / "lines.json", "w"), indent=1)
    json.dump([{"w": w, "t0": round(a, 3), "t1": round(e, 3), "line": k} for (k, w), a, e in zip(words, onsets, ends)], open(d / "words-ctc.json", "w"), indent=0)
    print(f"{len(words)} words in {len(meta['lines'])} lines; pause floor {floor:.0f} dB")
    for k, line in enumerate(meta["lines"]):
        print(f"{line['words'][0]['on']:7.2f}  " + " ".join(f"{x['w']}({x['off'] - x['on']:.2f})" for x in line["words"]))


if __name__ == "__main__":
    main(sys.argv[1])
