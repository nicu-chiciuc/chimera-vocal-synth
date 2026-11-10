"""Put known lyrics on Whisper's word timings: lyrics.py <words.json> <lyrics.txt> <out.json>

Whisper hears a dense rock vocal only roughly ("fuel and chesty" for "fuel-injected"), and the
synth sings whatever words the aligner is given. With the lyrics known, this matches them to
Whisper's words (difflib, on lowercase words without punctuation) and keeps Whisper's timing:
a matched word keeps its time; a run Whisper misheard gives its span to the lyric words, by
length; lyric words Whisper missed share the gap between their neighbours, or, with no gap, the
span of the word before them (MFA then times every
sound within its line anyway). Whisper's extra words are dropped, as is any lyric run with no
time to place it in (lines past the end of what Whisper heard). Prints what it changed.
"""
import difflib
import json
import re
import sys

norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'").replace("‘", "'")).strip("'")  # noqa: E731


def main():
    heard = [w for w in json.load(open(sys.argv[1])) if norm(w["w"])]
    # each lyric word and the lyric line it is on (prepare.py cut breaks lines there)
    lyric, line_of = [], {}
    for k, row in enumerate(r for r in open(sys.argv[2]).read().replace("-", " ").splitlines() if r.strip()):
        for w in row.split():
            if norm(w):
                line_of[len(lyric)] = k
                lyric.append(w)
    a, b = [norm(w["w"]) for w in heard], [norm(w) for w in lyric]
    out: list[dict] = []
    pending: list[str] = []  # lyric words waiting for a time

    def place(words, t0, t1):
        """words: (lyric index, word) pairs, spread over [t0, t1] by length"""
        if not words or t1 <= t0:
            return False
        lens = [max(1, len(norm(w))) for _, w in words]
        t = t0
        for (j, w), n in zip(words, lens):
            d = (t1 - t0) * n / sum(lens)
            out.append({"w": w, "t0": round(t, 3), "t1": round(t + d, 3), "line": line_of[j], "j": j})
            t += d
        return True

    ix = lambda j0, j1: [(j, lyric[j]) for j in range(j0, j1)]  # noqa: E731

    dropped, misheard = [], 0
    for op, i0, i1, j0, j1 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            if pending:
                # missed words go in the gap before this word; with no gap, they share the span
                # of the word before them (Whisper heard "till we" where he sings "till then
                # tramps like us, baby, we")
                # (just before it: a gap can be a whole instrumental break)
                prev_end = out[-1]["t1"] if out else 0.0
                if not place(pending, max(prev_end, heard[i0]["t0"] - 0.4 * len(pending)), heard[i0]["t0"]):
                    if out:
                        prev = out.pop()
                        if not place([(prev["j"], prev["w"])] + pending, prev["t0"], heard[i0]["t0"]):
                            out.append(prev)
                            dropped += [w for _, w in pending]
                    else:
                        dropped += [w for _, w in pending]
                pending = []
            for i, j in zip(range(i0, i1), range(j0, j1)):
                out.append({"w": lyric[j], "t0": heard[i]["t0"], "t1": heard[i]["t1"], "line": line_of[j], "j": j})
        elif op == "replace":
            misheard += 1
            print(f"heard {' '.join(w['w'] for w in heard[i0:i1])!r:46} -> {' '.join(lyric[j0:j1])!r}")
            if not place(pending + ix(j0, j1), heard[i0]["t0"], heard[i1 - 1]["t1"]):
                dropped += [w for _, w in pending + ix(j0, j1)]
            pending = []
        elif op == "insert":
            pending += ix(j0, j1)
        elif op == "delete":
            print(f"heard {' '.join(w['w'] for w in heard[i0:i1])!r:46} -> (not in the lyrics)")
    if pending:
        dropped += [w for _, w in pending]
    out.sort(key=lambda w: w["t0"])
    # a singer doesn't stop mid-line for an instrumental break: where Whisper put the start of a
    # line more than 3 s before its rest (it heard "The" of "The highway's jammed" 36 s early,
    # before the solo), the words before the hole move up to just before the rest
    moved = 0
    for k in sorted({w["line"] for w in out}):
        ws = [w for w in out if w["line"] == k]
        for i in range(len(ws) - 1, 0, -1):
            hole = ws[i]["t0"] - ws[i - 1]["t1"]
            if hole > 3.0:
                for w in ws[:i]:
                    w["t0"] = round(w["t0"] + hole - 0.05, 3)
                    w["t1"] = round(w["t1"] + hole - 0.05, 3)
                moved += i
    if moved:
        print(f"moved {moved} words across a break to the rest of their line")
    out.sort(key=lambda w: w["t0"])
    for w in out:
        del w["j"]
    json.dump(out, open(sys.argv[3], "w"), indent=0)
    print(f"{len(out)} words ({misheard} misheard runs fixed); no time for {len(dropped)}: {' '.join(dropped)[:200]}")


if __name__ == "__main__":
    main()
