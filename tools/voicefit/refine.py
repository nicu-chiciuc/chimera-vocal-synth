"""Repair the aligner's vowel-to-vowel boundaries in a track, from the singer's formants:

  refine.py <dir> [--dry]

The Montreal Forced Aligner is trained on speech. In singing it often puts the boundary between
two vowels in the wrong place when nothing but a light consonant separates them: in Hurt it gave
the opening "I" 70 ms (he holds it ~500 ms, F1 high and F2 gliding up, before F3 drops for the
"er" of "hurt") and gave the h of "make you hurt" 640 ms (the u of "you" runs on for 300 ms of it).
Every voice then sings that timing, so every voice gets the word wrong.

For each pair of vowels in a phrase with at most one light consonant between them (h, w, y, r, l):
an h over 150 ms keeps 60 ms and gives each vowel half the rest (an h can't hold a sung line,
where l, r, w and y can; formants can't place that boundary when the vowels sound alike, as
"you hurt" shows). Then, if either vowel
is under 120 ms (too short to sing: the aligner's mistake), it tries every boundary position (10 ms steps, each vowel at least 60 ms; the consonant keeps
min(its length, 60 ms) right after the boundary) and scores it by how well the singer's F1-F3 in
each part match that vowel's formants (the engine's male vowel table, scaled to the singer by the
median ratio over his vowels; a diphthong matches either of its two parts). A boundary moves only
if that cuts the pair's cost by 15 % and moves it 40 ms or more. Writes track.json (the aligner's
version kept as track-mfa.json) and prints every change.
"""
import json
import pathlib
import shutil
import sys

import librosa
import numpy as np
import parselmouth

REPO = pathlib.Path(__file__).resolve().parents[2]
STEP = 0.01
LIGHT = {"HH", "W", "Y", "R", "L"}
# ARPAbet vowels as the engine's vowel table's IPA (a diphthong: its two parts)
VOWELS = {
    "AA": ["ɑ"], "AE": ["æ"], "AH": ["ʌ", "ə"], "AO": ["ɔ"], "AW": ["a", "ʊ"], "AY": ["a", "ɪ"], "EH": ["ɛ"], "ER": ["ɝ"],
    "EY": ["e", "i"], "IH": ["ɪ"], "IY": ["i"], "OW": ["o", "u"], "OY": ["ɔ", "ɪ"], "UH": ["ʊ"], "UW": ["u"],
}


def base(p):
    return p.rstrip("012")


def is_vowel(p):
    return p[-1].isdigit()


def main():
    d = pathlib.Path(sys.argv[1])
    dry = "--dry" in sys.argv
    track = json.load(open(d / "track.json"))
    table = json.load(open(REPO / "crates/chimera/src/phonetics/vowelTargets.generated.json"))["male"]
    y, sr = librosa.load(d / "target.wav", sr=16000)
    snd = parselmouth.Sound(y, sr)
    fm = snd.to_formant_burg(time_step=STEP, max_number_of_formants=5, maximum_formant=5000)
    n = int(snd.duration / STEP)
    F = np.array([[fm.get_value_at_time(k, i * STEP) for k in (1, 2, 3)] for i in range(n)])
    hz = np.array(track["f0"]["hz"])
    hop = track["f0"]["hop"]
    voiced = np.array([hz[min(len(hz) - 1, int(i * STEP / hop))] > 0 for i in range(n)]) & np.all(np.isfinite(F), 1)
    # scale the table to this singer: median ratio of his formants to the table's over his vowels
    pairs = []
    for ph in track["phrases"]:
        for w in ph["words"]:
            for p, a, b in w["phones"]:
                if is_vowel(p) and len(VOWELS[base(p)]) == 1:
                    for i in range(int(a / STEP), int(b / STEP)):
                        if i < n and voiced[i]:
                            pairs.append(F[i] / np.array(table[VOWELS[base(p)][0]]["formants"][:3]))
    scale = np.median(np.array(pairs), 0)
    print(f"singer / table formant ratio: F1 {scale[0]:.2f}, F2 {scale[1]:.2f}, F3 {scale[2]:.2f}")
    targets = {v: [np.array(table[x]["formants"][:3]) * scale for x in xs] for v, xs in VOWELS.items()}

    def cost(v, i0, i1):
        """how badly frames [i0, i1) match vowel v (mean over voiced frames, log-formant distance)"""
        idx = np.arange(max(0, i0), min(n, i1))
        idx = idx[voiced[idx]]
        if len(idx) == 0:
            return 0.0, 0
        lf = np.log(F[idx])
        best = np.min([np.abs(lf - np.log(t)).sum(1) for t in targets[base(v)]], 0)
        return float(best.sum()), len(idx)

    changes = 0
    for ph in track["phrases"]:
        flat = [x for w in ph["words"] for x in w["phones"]]
        for i in range(len(flat) - 1):
            v1 = flat[i]
            if not is_vowel(v1[0]):
                continue
            mid = None
            j = i + 1
            if not is_vowel(flat[j][0]):
                if base(flat[j][0]) not in LIGHT or j + 1 >= len(flat) or not is_vowel(flat[j + 1][0]):
                    continue
                mid, j = flat[j], j + 1
            v2 = flat[j]
            # an h can't hold a sung line (l, r, w, y can: a singer sustains them): past 150 ms the
            # time is the vowels', so it keeps 60 ms and each vowel gets half the rest
            if mid and base(mid[0]) == "HH" and mid[2] - mid[1] > 0.15:
                extra = (mid[2] - mid[1] - 0.06) / 2
                print(f"{ph['text'][:28]:28s} {v1[0]}|{mid[0]}|{v2[0]}: {mid[0]} was {mid[2] - mid[1]:.2f} s; now 0.06 s at {mid[1] + extra:.2f}")
                v1[2] = round(mid[1] + extra, 3)
                mid[1], mid[2] = v1[2], round(v1[2] + 0.06, 3)
                v2[1] = mid[2]
                changes += 1
            # only a pair with a vowel too short to sing is suspect
            if min(v1[2] - v1[1], v2[2] - v2[1]) >= 0.12:
                continue
            lo, hi = v1[1], v2[2]
            c = min(0.06, mid[2] - mid[1]) if mid else 0.0

            def total(b):
                c1, n1 = cost(v1[0], int(lo / STEP), int(b / STEP))
                c2, n2 = cost(v2[0], int((b + c) / STEP), int(hi / STEP))
                return (c1 + c2) / max(1, n1 + n2)

            old_b = v1[2]
            old = total(old_b)
            cands = np.arange(lo + 0.06, hi - c - 0.06, STEP)
            if len(cands) == 0:
                continue
            scores = [total(b) for b in cands]
            k = int(np.argmin(scores))
            new_b = float(cands[k])
            if abs(new_b - old_b) >= 0.04 and scores[k] < 0.85 * old:
                changes += 1
                print(f"{ph['text'][:28]:28s} {v1[0]}|{mid[0] if mid else ''}|{v2[0]}: boundary {old_b:.2f} -> {new_b:.2f} s (cost {old:.3f} -> {scores[k]:.3f})")
                v1[2] = round(new_b, 3)
                if mid:
                    mid[1], mid[2] = round(new_b, 3), round(new_b + c, 3)
                v2[1] = round(new_b + c, 3)
    print(f"{changes} boundaries moved")
    if not dry and changes:
        if not (d / "track-mfa.json").exists():
            shutil.copy(d / "track.json", d / "track-mfa.json")
        json.dump(track, open(d / "track.json", "w"))


if __name__ == "__main__":
    main()
