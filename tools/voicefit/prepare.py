"""Turn a sung vocal into a target for `chimera render track` and tools/voicefit/fit.py.

  prepare.py cut <vocal.wav> <words.json> <t0> <t1> <dir>
      dir/target.wav (48 kHz mono, the excerpt), dir/lines.json (Whisper's words grouped into
      lines), and an MFA corpus in dir/mfa/corpus, one line per utterance.
  prepare.py backing <no_vocals.wav> <dir>
      dir/accompaniment.wav: the separated instrumental (Demucs' no_vocals), cut to the same
      excerpt as target.wav, so the comparison page can play every rendition over the band.
  prepare.py collect <dir> [floor ceiling]
      reads MFA's JSON in dir/mfa/aligned and writes dir/track.json: the phrases (one per line,
      phones with their times on the excerpt's clock) and the singer's pitch curve (Praat, 5 ms,
      searched between floor and ceiling Hz: 55-320 by default, for a low voice like Cash; a
      tenor needs a higher floor, or Praat follows the bass that leaks into the vocal stem).

The vocal's audio stays in the dataset directory, never in git.
"""
import json
import pathlib
import sys

import librosa
import numpy as np
import parselmouth
import soundfile as sf

FS = 48000
HOP = 0.005
MAX_LINE = 10.0  # past this a line breaks at the next short pause (a singer who hardly stops)
SHORT_GAP = 0.12
LINE_GAP = 0.5  # a pause this long between words starts a new line
PAD = 0.3  # audio around each line for the aligner


def cut(vocal, words_path, t0, t1, out):
    out = pathlib.Path(out)
    (out / "mfa" / "corpus").mkdir(parents=True, exist_ok=True)
    y, _ = librosa.load(vocal, sr=FS, mono=True, offset=t0, duration=t1 - t0)
    sf.write(out / "target.wav", y, FS, subtype="FLOAT")
    words = [w for w in json.load(open(words_path)) if w["t0"] >= t0 and w["t1"] <= t1]
    lines = []
    for w in words:
        w = {"w": w["w"], "t0": round(w["t0"] - t0, 3), "t1": round(w["t1"] - t0, 3), **({"line": w["line"]} if "line" in w else {})}
        gap = w["t0"] - lines[-1][-1]["t1"] if lines else 0.0
        # known lyrics (lyrics.py) carry their line: break there, and only there
        if "line" in w:
            new = not lines or w["line"] != lines[-1][-1]["line"]
        else:
            new = not lines or gap > LINE_GAP or (w["t1"] - lines[-1][0]["t0"] > MAX_LINE and gap > SHORT_GAP)
        if new:
            lines.append([])
        lines[-1].append(w)
    y16 = librosa.resample(y, orig_sr=FS, target_sr=16000)
    spans = []
    for i, line in enumerate(lines):
        a = max(0.0, line[0]["t0"] - PAD)
        b = min(len(y) / FS, line[-1]["t1"] + PAD)
        name = f"line{i + 1:02d}"
        d = out / "mfa" / "corpus" / name
        d.mkdir(exist_ok=True)
        sf.write(d / f"{name}.wav", y16[int(a * 16000) : int(b * 16000)], 16000)
        # the aligner's dictionary spells with straight apostrophes and no outer quotes ("‘cause")
        text = " ".join(w["w"].replace("’", "'").replace("‘", "'").strip(",.?!;:'\"").lower() for w in line)
        (d / f"{name}.lab").write_text(text + "\n")
        spans.append({"name": name, "t0": round(a, 3), "t1": round(b, 3), "text": text})
    json.dump({"source": str(vocal), "t0": t0, "t1": t1, "lines": spans}, open(out / "lines.json", "w"), indent=1)
    print(f"{len(lines)} lines, {t1 - t0:.1f} s")


def backing(no_vocals, out):
    out = pathlib.Path(out)
    meta = json.load(open(out / "lines.json"))
    y, _ = librosa.load(no_vocals, sr=FS, mono=True, offset=meta["t0"], duration=meta["t1"] - meta["t0"])
    sf.write(out / "accompaniment.wav", y, FS, subtype="FLOAT")
    print(f"{out / 'accompaniment.wav'}: {len(y) / FS:.1f} s")


def collect(out, floor=55.0, ceiling=320.0):
    out = pathlib.Path(out)
    meta = json.load(open(out / "lines.json"))
    phrases = []
    missing = []
    for line in meta["lines"]:
        # one utterance per line (cut), or one per word (align.py), each with its own clock
        utts = line.get("words") or [line]
        ws = []
        for u in utts:
            try:
                j = json.load(open(out / "mfa" / "aligned" / u["name"] / f"{u['name']}.json"))
            except FileNotFoundError:
                missing.append(u["name"])
                ws.append({"w": u["w"], "phones": spread(u["w"], u["on"], u["off"])})
                continue
            tiers = j["tiers"]
            words = [e for e in tiers["words"]["entries"] if e[2] not in ("", "<eps>", "spn", "sil")]
            phones = [e for e in tiers["phones"]["entries"] if e[2] not in ("", "sil", "sp", "spn")]
            for a, b, w in words:
                ps = [[p, round(u["t0"] + pa, 3), round(u["t0"] + pb, 3)] for pa, pb, p in phones if pa >= a - 1e-6 and pb <= b + 1e-6]
                if "off" in u:
                    ps = hold(ps, u["on"], u["off"])
                ws.append({"w": w, "phones": ps})
        # a word too short to sing ("I" in 40 ms) grows to MIN_WORD, backwards into the pause
        # before it where there is one, else forwards over the next word
        for k, w in enumerate(ws):
            if not w["phones"]:
                continue
            a, b = w["phones"][0][1], w["phones"][-1][2]
            if b - a < MIN_WORD:
                prev_end = ws[k - 1]["phones"][-1][2] if k and ws[k - 1]["phones"] else 0.0
                na = max(prev_end, b - MIN_WORD)
                w["phones"] = rescale(w["phones"], na, max(b, na + MIN_WORD))
        # words aligned one by one can overlap a little: the longer one gives way
        for prev, nxt in zip(ws, ws[1:]):
            if prev["phones"] and nxt["phones"] and prev["phones"][-1][2] > nxt["phones"][0][1]:
                p0, p1, n0, n1 = prev["phones"][0][1], prev["phones"][-1][2], nxt["phones"][0][1], nxt["phones"][-1][2]
                if p1 - p0 <= n1 - n0:
                    nxt["phones"] = rescale(nxt["phones"], p1, max(n1, p1 + MIN_WORD))
                else:
                    prev["phones"] = rescale(prev["phones"], min(p0, n0 - MIN_WORD), n0)
        phrases.append({"text": line["text"], "words": ws})
    if missing:
        print(f"MFA gave nothing for {len(missing)} words (phones spread by rule): {' '.join(missing)}")
    snd = parselmouth.Sound(str(out / "target.wav"))
    pitch = snd.to_pitch_ac(time_step=HOP, pitch_floor=floor, pitch_ceiling=ceiling, very_accurate=True)
    n = int(snd.duration / HOP)
    hz = [0.0] * n
    for i in range(n):
        f = pitch.get_value_at_time(i * HOP)
        hz[i] = round(float(f), 2) if f == f else 0.0
    hz = fix_octaves(hz)
    track = {
        "body": "male",
        "duration": round(snd.duration, 3),
        "phrases": phrases,
        "f0": {"hop": HOP, "hz": hz},
    }
    json.dump(track, open(out / "track.json", "w"))
    voiced = np.array([h for h in hz if h > 0])
    print(f"{len(phrases)} phrases, {sum(len(w['phones']) for p in phrases for w in p['words'])} phones; "
          f"f0 median {np.median(voiced):.0f} Hz, 5–95 % {np.percentile(voiced, 5):.0f}–{np.percentile(voiced, 95):.0f} Hz")


DICT = pathlib.Path.home() / ".cache/chimera-mfa/mfa/pretrained_models/dictionary/english_us_arpa.dict"
VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())


HOLD_SLACK = 0.15  # a word MFA starts or ends this close to its CTC span is left as MFA placed it
MIN_WORD = 0.08  # the shortest word the synth can get out


def rescale(phones, t0, t1):
    """The phones laid over [t0, t1] in proportion."""
    a, b = phones[0][1], phones[-1][2]
    s = (t1 - t0) / max(b - a, 1e-3)
    return [[p, round(t0 + (x - a) * s, 3), round(t0 + (y - a) * s, 3)] for p, x, y in phones]


def hold(phones, on, off):
    """Put a word's phones on its sung span [on, off] (align.py: the CTC aligner's onset, and
    the next onset or a pause). MFA squeezes a held word's phones into 0.3 s and calls the
    sustain silence ("roads" held 2.4 s came out as R OW D Z in 0.32 s), sometimes at the end
    of the sustain instead of its start. So: a word that starts late moves to its onset, keeping
    its phone lengths; then the last vowel takes the rest of the span, and the consonants after
    that vowel keep their lengths and move to the end of the span."""
    if not phones:
        return phones
    if phones[0][1] > on + HOLD_SLACK:
        shift = phones[0][1] - on
        phones = [[p, round(a - shift, 3), round(b - shift, 3)] for p, a, b in phones]
    if phones[-1][2] >= off - HOLD_SLACK:
        return phones
    vowels = [i for i, (p, _, _) in enumerate(phones) if p.rstrip("012") in VOWELS]
    i = vowels[-1] if vowels else len(phones) - 1
    tail = sum(b - a for _, a, b in phones[i + 1 :])
    out = phones[:i] + [[phones[i][0], phones[i][1], round(off - tail, 3)]]
    t = off - tail
    for p, a, b in phones[i + 1 :]:
        out.append([p, round(t, 3), round(t + (b - a), 3)])
        t += b - a
    return out


def spread(word, t0, t1, consonant=0.06):
    """Phones of a word the aligner gave up on, from the dictionary's first pronunciation:
    consonants 60 ms each, the vowels sharing the rest."""
    pron = None
    for row in open(DICT, encoding="utf-8"):
        cols = row.split()
        if cols and cols[0] == word.lower():
            pron = [c for c in cols[1:] if c[0].isupper()]  # after the word: probabilities, then phones
            break
    if not pron:
        return [["AH1", round(t0, 3), round(t1, 3)]]
    nv = sum(p.rstrip("012") in VOWELS for p in pron) or 1
    c = min(consonant, (t1 - t0) / (2 * len(pron)))
    v = ((t1 - t0) - c * (len(pron) - nv)) / nv
    out, t = [], t0
    for p in pron:
        d = v if p.rstrip("012") in VOWELS else c
        out.append([p, round(t, 3), round(t + d, 3)])
        t += d
    return out


def fix_octaves(hz, half_window=40):
    """Move frames an octave off their neighbourhood back: in creak Praat often reads half the
    pitch (or double) for 50-100 ms, and the synth would sing that as a sudden drop. The
    neighbourhood is the median of the voiced frames within 0.2 s either side."""
    hz = np.array(hz, float)
    v = np.flatnonzero(hz > 0)
    out = hz.copy()
    for i in v:
        near = hz[max(0, i - half_window) : i + half_window + 1]
        ref = np.median(near[near > 0])
        r = np.log2(hz[i] / ref)
        if r < -0.7:
            out[i] = hz[i] * 2
        elif r > 0.7:
            out[i] = hz[i] / 2
    print(f"octave-corrected {(out != hz).sum()} of {len(v)} voiced frames")
    return [round(float(x), 2) for x in out]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cut":
        cut(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
    elif cmd == "backing":
        backing(sys.argv[2], sys.argv[3])
    elif cmd == "collect":
        collect(sys.argv[2], *map(float, sys.argv[3:5]))
    else:
        sys.exit(__doc__)
