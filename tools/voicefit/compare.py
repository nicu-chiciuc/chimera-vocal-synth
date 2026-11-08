"""How far a render of `chimera render track` is from the recording it copies.

  compare.py <target.wav> <synth.wav> <track.json> [--phrases 0-3]

Both files share one timeline (the render sings the track's aligned phones at the singer's pitch),
so frames and phones line up and every measure compares like with like. Lower is better; each
term is divided by a unit of roughly one clearly audible step, and the score is the mean of six
groups:

  envelope   RMS dB difference of the two spectral envelopes over speech frames (40 mel bands to
             8 kHz, smoothed to cepstral coefficients c1-c13, so level and harmonics drop out):
             the overall sound of each moment
  tilt       band levels re the 0-8 kHz total over vowels (0-0.5, 0.5-1, 1-2, 2-4, 4-8 kHz)
  formants   F1-F3 per vowel token (Praat Burg), median |log ratio|
  source     per vowel token: H1-H2, CPPS, HNR, jitter, shimmer (medians over tokens)
  loudness   correlation of the two level contours over speech frames, each floored 25 dB under
             its median vowel level (a stop's closure is silent in the synth and holds the
             room's reverb in a recording; that difference is the room, not the voice)
  voicing    share of speech frames whose voiced/unvoiced call agrees

The measures and units are fixed: a fit may not change them to win.
"""
import json
import sys

import librosa
import numpy as np
import parselmouth
import scipy.fft
from parselmouth.praat import call

FS = 48000
HOP = 240  # 5 ms
WIN = 1200  # 25 ms
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
UNITS = {
    "envelope": 2.0,  # dB
    "band": 3.0,  # dB
    "formant": 0.06,  # |log ratio| (about 6 %)
    "h1h2": 3.0,  # dB
    "cpps": 2.0,  # dB
    "hnr": 3.0,  # dB
    "jitter": 0.5,  # % points
    "shimmer": 2.0,  # % points
    "loudness": 0.05,  # 1 - r
    "voicing": 0.05,  # disagreeing share
}


def phones(track, which=None):
    """(arpa, t0, t1) of the chosen phrases' phones."""
    out = []
    for i, p in enumerate(track["phrases"]):
        if which is not None and i not in which:
            continue
        for w in p["words"]:
            out.extend((a, t0, t1) for a, t0, t1 in w["phones"])
    return out


def is_vowel(arpa):
    return arpa[-1].isdigit()


def frames_in(spans, n):
    m = np.zeros(n, bool)
    for t0, t1 in spans:
        m[int(t0 * FS / HOP) : int(t1 * FS / HOP)] = True
    return m


class Analysis:
    """Everything measured on one file, over the chosen phones."""

    def __init__(self, path, track, which, f0_track=None):
        ph = phones(track, which)
        # analyse only the chosen phrases (times stay on the track's clock: frame 0 is time 0)
        start = max(0.0, min(a for _, a, _ in ph) - 0.1)
        y, _ = librosa.load(path, sr=FS, mono=True, duration=max(b for _, _, b in ph) + 0.1)
        y[: int(start * FS)] = 0.0
        self.y = y
        n = 1 + len(y) // HOP
        self.speech = frames_in([(a, b) for _, a, b in ph], n)
        vowel_mid = [(a + 0.2 * (b - a), b - 0.2 * (b - a)) for p, a, b in ph if is_vowel(p)]
        self.vowel = frames_in(vowel_mid, n)
        spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP, win_length=WIN)) ** 2
        n = min(n, spec.shape[1])
        self.speech, self.vowel, spec = self.speech[:n], self.vowel[:n], spec[:, :n]
        mel = librosa.feature.melspectrogram(S=spec, sr=FS, n_mels=40, fmax=8000)
        db = 10 * np.log10(np.maximum(mel, 1e-12))
        cep = scipy.fft.dct(db, type=2, norm="ortho", axis=0)
        cep[0] = 0.0
        cep[14:] = 0.0
        self.env = scipy.fft.idct(cep, type=2, norm="ortho", axis=0)  # dB, level removed
        self.level = 10 * np.log10(np.maximum(spec.sum(0), 1e-12))
        freqs = librosa.fft_frequencies(sr=FS, n_fft=2048)
        ltas = spec[:, self.vowel].mean(1)
        total = ltas[(freqs >= 0) & (freqs < 8000)].sum()
        self.bands = [10 * np.log10(ltas[(freqs >= a) & (freqs < b)].sum() / total) for a, b in BANDS]
        snd = parselmouth.Sound(y, FS)
        pitch = snd.to_pitch_ac(time_step=HOP / FS, pitch_floor=55, pitch_ceiling=320)
        ts = np.arange(n) * HOP / FS
        self.voiced = np.array([pitch.get_value_at_time(t) for t in ts])
        self.voiced = np.nan_to_num(self.voiced) > 0
        snd = snd.extract_part(from_time=start, to_time=snd.duration, preserve_times=True)
        fm = snd.to_formant_burg(time_step=HOP / FS, max_number_of_formants=5, maximum_formant=5000)
        self.tokens = []
        hz = np.array(f0_track["hz"])
        for p, a, b in ph:
            if not is_vowel(p) or b - a < 0.08:
                continue
            m0, m1 = a + 0.25 * (b - a), b - 0.25 * (b - a)
            f0s = hz[int(m0 / f0_track["hop"]) : int(m1 / f0_track["hop"])]
            f0s = f0s[f0s > 0]
            if len(f0s) < 3:
                continue
            tok = {"vowel": p.rstrip("012"), "t": (m0, m1)}
            mids = np.linspace(m0, m1, 7)
            for k in (1, 2, 3):
                v = [call(fm, "Get value at time", k, t, "hertz", "Linear") for t in mids]
                tok[f"F{k}"] = float(np.nanmedian(v))
            tok["h1h2"] = h1h2(y, m0, m1, float(np.median(f0s)))
            part = snd.extract_part(from_time=a, to_time=b, preserve_times=False)
            tok["cpps"] = cpp(y, m0, m1)
            harm = call(part, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0)
            tok["hnr"] = call(harm, "Get mean", 0, 0)
            pp = call(part, "To PointProcess (periodic, cc)", 60, 320)
            tok["jitter"] = 100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            tok["shimmer"] = 100 * call([part, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            self.tokens.append(tok)


def h1h2(y, t0, t1, f0):
    """First minus second harmonic (dB) over [t0, t1] at pitch f0."""
    x = y[int(t0 * FS) : int(t1 * FS)]
    if len(x) < 256:
        return np.nan
    x = x * np.hanning(len(x))
    nfft = 1 << int(np.ceil(np.log2(len(x) * 4)))
    s = np.abs(np.fft.rfft(x, nfft))
    f = np.fft.rfftfreq(nfft, 1 / FS)

    def peak(h):
        m = (f > h * f0 * 0.9) & (f < h * f0 * 1.1)
        return 20 * np.log10(s[m].max() + 1e-12)

    return peak(1) - peak(2)


def cpp(y, t0, t1):
    """Cepstral peak prominence (dB, Hillenbrand 1994) averaged over 40 ms frames in [t0, t1]:
    the height of the pitch peak in the power cepstrum above its regression line. Low is breathy
    or rough, high is clean."""
    n = int(0.04 * FS)
    out = []
    for s in range(int(t0 * FS), int(t1 * FS) - n, n // 2):
        x = y[s : s + n] * np.hanning(n)
        spec = 20 * np.log10(np.abs(np.fft.rfft(x, 4096)) + 1e-9)
        ceps = 20 * np.log10(np.abs(np.fft.irfft(spec)) + 1e-9)[: 4096 // 2]
        q = np.arange(len(ceps)) / FS
        m = (q >= 1 / 320) & (q <= 1 / 60)
        i = np.argmax(ceps[m])
        slope, icpt = np.polyfit(q[m], ceps[m], 1)
        out.append(ceps[m][i] - (slope * q[m][i] + icpt))
    return float(np.mean(out)) if out else np.nan


def nanmed(xs):
    xs = [x for x in xs if x == x]
    return float(np.median(xs)) if xs else np.nan


FLOOR = 25  # dB under the median vowel level: below it a recording holds its room, not the voice


def floored(a, n):
    return np.maximum(a.level[:n], np.median(a.level[:n][a.vowel[:n]]) - FLOOR)


def compare(tgt, syn):
    n = min(len(tgt.level), len(syn.level))
    lt, ls = floored(tgt, n), floored(syn, n)
    # the envelope only where the singer is clearly sounding (a stop's closure in the recording is
    # reverb and separation residue)
    sp = tgt.speech[:n] & (lt > np.median(lt[tgt.vowel[:n]]) - FLOOR)
    d = tgt.env[:, :n][:, sp] - syn.env[:, :n][:, sp]
    env = float(np.mean(np.sqrt((d**2).mean(0))))
    r = float(np.corrcoef(lt[tgt.speech[:n]], ls[tgt.speech[:n]])[0, 1])
    agree = float(np.mean(tgt.voiced[:n][tgt.speech[:n]] == syn.voiced[:n][tgt.speech[:n]]))
    terms = {"envelope dB": (None, env, env / UNITS["envelope"])}
    for (a, b), x, z in zip(BANDS, tgt.bands, syn.bands):
        terms[f"band {a / 1000:g}-{b / 1000:g}k"] = (x, z, abs(z - x) / UNITS["band"])
    pairs = [(t, s) for t, s in zip(tgt.tokens, syn.tokens) if t["t"] == s["t"]]
    for k in ("F1", "F2", "F3"):
        e = [abs(np.log(s[k] / t[k])) for t, s in pairs if t[k] > 0 and s[k] > 0]
        terms[k] = (nanmed([t[k] for t, _ in pairs]), nanmed([s[k] for _, s in pairs]), (float(np.median(e)) if e else 1.0) / UNITS["formant"])
    for k in ("h1h2", "cpps", "hnr", "jitter", "shimmer"):
        x, z = nanmed([t[k] for t, _ in pairs]), nanmed([s[k] for _, s in pairs])
        terms[k] = (x, z, abs(z - x) / UNITS[k] if x == x and z == z else 3.0)
    terms["loudness r"] = (1.0, r, (1 - r) / UNITS["loudness"])
    terms["voicing agree"] = (1.0, agree, (1 - agree) / UNITS["voicing"])
    groups = {
        "envelope": ["envelope dB"],
        "tilt": [k for k in terms if k.startswith("band")],
        "formants": ["F1", "F2", "F3"],
        "source": ["h1h2", "cpps", "hnr", "jitter", "shimmer"],
        "loudness": ["loudness r"],
        "voicing": ["voicing agree"],
    }
    gcost = {g: float(np.mean([terms[k][2] for k in ks])) for g, ks in groups.items()}
    return float(np.mean(list(gcost.values()))), terms, gcost


def report(score, terms, gcost):
    lines = [f"score {score:.3f}   " + "  ".join(f"{g} {c:.2f}" for g, c in gcost.items())]
    for k, (x, z, c) in terms.items():
        xs = "" if x is None else f"{x:8.2f}"
        lines.append(f"  {k:14s} target {xs:>8s}  synth {z:8.2f}  cost {c:5.2f}")
    return "\n".join(lines)


def parse_which(s):
    if s is None:
        return None
    a, _, b = s.partition("-")
    return set(range(int(a), int(b or a) + 1))


if __name__ == "__main__":
    args = sys.argv[1:]
    which = None
    if "--phrases" in args:
        i = args.index("--phrases")
        which = parse_which(args[i + 1])
        del args[i : i + 2]
    track = json.load(open(args[2]))
    tgt = Analysis(args[0], track, which, track["f0"])
    syn = Analysis(args[1], track, which, track["f0"])
    print(report(*compare(tgt, syn)))
