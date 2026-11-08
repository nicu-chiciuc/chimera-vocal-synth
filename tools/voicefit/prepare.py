"""Turn a sung vocal into a target for `chimera render track` and tools/voicefit/fit.py.

  prepare.py cut <vocal.wav> <words.json> <t0> <t1> <dir>
      dir/target.wav (48 kHz mono, the excerpt), dir/lines.json (Whisper's words grouped into
      lines), and an MFA corpus in dir/mfa/corpus, one line per utterance.
  prepare.py backing <no_vocals.wav> <dir>
      dir/accompaniment.wav: the separated instrumental (Demucs' no_vocals), cut to the same
      excerpt as target.wav, so the comparison page can play every rendition over the band.
  prepare.py collect <dir>
      reads MFA's JSON in dir/mfa/aligned and writes dir/track.json: the phrases (one per line,
      phones with their times on the excerpt's clock) and the singer's pitch curve (Praat, 5 ms).

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
        w = {"w": w["w"], "t0": round(w["t0"] - t0, 3), "t1": round(w["t1"] - t0, 3)}
        if not lines or w["t0"] - lines[-1][-1]["t1"] > LINE_GAP:
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
        text = " ".join(w["w"].strip(",.?!").lower() for w in line)
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


def collect(out):
    out = pathlib.Path(out)
    meta = json.load(open(out / "lines.json"))
    phrases = []
    for line in meta["lines"]:
        j = json.load(open(out / "mfa" / "aligned" / line["name"] / f"{line['name']}.json"))
        tiers = j["tiers"]
        words = [e for e in tiers["words"]["entries"] if e[2] not in ("", "<eps>", "spn", "sil")]
        phones = [e for e in tiers["phones"]["entries"] if e[2] not in ("", "sil", "sp", "spn")]
        ws = []
        for a, b, w in words:
            ps = [[p, round(line["t0"] + pa, 3), round(line["t0"] + pb, 3)] for pa, pb, p in phones if pa >= a - 1e-6 and pb <= b + 1e-6]
            ws.append({"w": w, "phones": ps})
        phrases.append({"text": line["text"], "words": ws})
    snd = parselmouth.Sound(str(out / "target.wav"))
    pitch = snd.to_pitch_ac(time_step=HOP, pitch_floor=55, pitch_ceiling=320, very_accurate=True)
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
        collect(sys.argv[2])
    else:
        sys.exit(__doc__)
