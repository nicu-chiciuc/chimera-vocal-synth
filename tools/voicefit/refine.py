"""Repair the aligner's vowel-to-vowel boundaries in a track, from the singer's formants:

  refine.py <dir> [--dry] [--onsets-only]

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
HELD = 0.5  # a vowel this long is a held note
HELD_MARGIN = 0.15  # another vowel of the word has to fit the sustain's formants this much better
# ARPAbet vowels as the engine's vowel table's IPA (a diphthong: its two parts)
VOWELS = {
    "AA": ["ɑ"], "AE": ["æ"], "AH": ["ʌ", "ə"], "AO": ["ɔ"], "AW": ["a", "ʊ"], "AY": ["a", "ɪ"], "EH": ["ɛ"], "ER": ["ɝ"],
    "EY": ["e", "i"], "IH": ["ɪ"], "IY": ["i"], "OW": ["o", "u"], "OY": ["ɔ", "ɪ"], "UH": ["ʊ"], "UW": ["u"],
}


def base(p):
    return p.rstrip("012")


def is_vowel(p):
    return p[-1].isdigit()


def recover_vowel_onsets(track, db, voiced):
    """Recover a clipped phrase-initial vowel from the adjacent voiced reference audio.

    Only vowels under 120 ms qualify. Look back at most 350 ms, never into the previous
    phrase, and stop at a 30 ms break. Require 60 ms of voiced evidence before extending.
    The vowel's end and every following phone stay fixed.
    """
    changes, previous_end = 0, 0.0
    for phrase in track["phrases"]:
        phones = [p for w in phrase["words"] for p in w["phones"]]
        if not phones:
            continue
        first = phones[0]
        lower = max(previous_end, first[1] - 0.35)
        previous_end = max(previous_end, max(p[2] for p in phones))
        if not is_vowel(first[0]) or first[2] - first[1] >= 0.12:
            continue
        start, end = int(first[1] / STEP), int(np.ceil(first[2] / STEP))
        end = min(end, len(db), len(voiced))
        if start >= end:
            continue
        floor = float(np.max(db[start:end])) - 18.0
        earliest, gap, evidence = start, 0, 0
        for i in range(start - 1, int(np.ceil(lower / STEP)) - 1, -1):
            if voiced[i] and db[i] >= floor:
                earliest, gap, evidence = i, 0, evidence + 1
            else:
                gap += 1
                if gap >= 3:
                    break
        if evidence >= 6 and first[1] - earliest * STEP >= 0.06:
            onset = round(earliest * STEP, 3)
            print(f"{phrase['text'][:28]:28s} {first[0]} onset {first[1]:.3f} -> {onset:.3f} s")
            first[1] = onset
            changes += 1
    return changes


def save_track(d, track, dry, changes):
    if not dry and changes:
        if not (d / "track-mfa.json").exists():
            shutil.copy(d / "track.json", d / "track-mfa.json")
        json.dump(track, open(d / "track.json", "w"))


def main():
    d = pathlib.Path(sys.argv[1])
    dry = "--dry" in sys.argv
    track = json.load(open(d / "track.json"))
    y, sr = librosa.load(d / "target.wav", sr=16000)
    hz = np.array(track["f0"]["hz"])
    hop = track["f0"]["hop"]
    n = int(len(y) / sr / STEP)
    pitch_voiced = np.array([hz[min(len(hz) - 1, int(i * STEP / hop))] > 0 for i in range(n)])
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=int(sr * STEP))[0][:n]
    changes = recover_vowel_onsets(track, 20 * np.log10(rms + 1e-9), pitch_voiced)
    if "--onsets-only" in sys.argv:
        save_track(d, track, dry, changes)
        print(f"{changes} vowel onsets recovered")
        return
    table = json.load(open(REPO / "crates/chimera/src/phonetics/vowelTargets.generated.json"))["male"]
    snd = parselmouth.Sound(y, sr)
    fm = snd.to_formant_burg(time_step=STEP, max_number_of_formants=5, maximum_formant=5000)
    n = int(snd.duration / STEP)
    F = np.array([[fm.get_value_at_time(k, i * STEP) for k in (1, 2, 3)] for i in range(n)])
    voiced = pitch_voiced[:n] & np.all(np.isfinite(F), 1)
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
    # the held note of a word with several vowels: the sustain's formants say which vowel he
    # sings it on. The aligner's rule (prepare.py hold) gives it to the last vowel, right for
    # "Virgin-ia", "be-long" and "moun-tains"; but "mama" is held on an open ah, not the
    # dictionary's schwa, and the long "country" of the outro on "coun-", not "-try"
    held = 0
    for ph in track["phrases"]:
        for w in ph["words"]:
            vs = [i for i, p in enumerate(w["phones"]) if is_vowel(p[0])]
            if len(vs) < 2:
                continue
            h = max(vs, key=lambda i: w["phones"][i][2] - w["phones"][i][1])
            hp = w["phones"][h]
            if hp[2] - hp[1] < HELD:
                continue
            i0, i1 = int((hp[1] + 0.15) / STEP), int((hp[2] - 0.1) / STEP)
            costs = {}
            for i in vs:
                c, k = cost(w["phones"][i][0], i0, i1)
                if k >= 5:
                    costs[i] = c / k
            if h not in costs:
                continue
            b = min(costs, key=costs.get)
            if b == h or costs[h] - costs[b] < HELD_MARGIN:
                continue
            if base(hp[0]) == "AH" and hp[0].endswith("0"):
                # a schwa opens up when held: sing it as the vowel heard
                print(f"{ph['text'][:28]:28s} {w['w']}: held {hp[0]} sung as {w['phones'][b][0]}")
                hp[0] = w["phones"][b][0]
            elif b < h:
                # the note is held earlier in the word: the phones after that vowel pack at its end
                tail = w["phones"][b + 1 :]
                durs = [min(0.12, p[2] - p[1]) for p in tail]
                t = hp[2] - sum(durs)
                if t < w["phones"][b][1] + 0.1:
                    continue
                print(f"{ph['text'][:28]:28s} {w['w']}: the hold moves from {hp[0]} to {w['phones'][b][0]}")
                w["phones"][b][2] = round(t, 3)
                for p, dd in zip(tail, durs):
                    p[1], p[2] = round(t, 3), round(t + dd, 3)
                    t += dd
            else:
                continue
            held += 1
    print(f"{held} held vowels changed")
    changes += held
    save_track(d, track, dry, changes)


if __name__ == "__main__":
    main()
