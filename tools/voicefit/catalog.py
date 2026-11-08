"""Collect a fit's candidates for the comparison page (the app's `?voicefit`):

  catalog.py <dir> [--train 0-3] [--hold 4-7]

Writes <dir>/candidates.json: the singer's lines (text, times, training or held-out) and every
candidate voice: the untuned default (fit/default-full.wav) and each <dir>/fit*/ with a best.json
(its best-full.wav, rendered if missing). For each: the settings, the acoustic score per split
(compare.py), and the independent judges per split (judge.py: speaker similarity and Whisper's
word error). Scores and judges are cached in <candidate dir>/catalog-cache.json, keyed by the
render's size and time, so adding a fit only measures the new one.

A candidate's label is the first line of <fit dir>/about.txt (or the folder's name); the rest of
that file is its description. The app's voice profile made from it (profile.py) is linked by the
profile's `fit` field.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compare  # noqa: E402
import fit  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


def stamp(p):
    s = os.stat(p)
    return f"{s.st_size}:{int(s.st_mtime)}"


def main():
    args = sys.argv[1:]
    d = pathlib.Path(args[0]).resolve()
    opt = {"--train": "0-3", "--hold": "4-7"}
    for k in opt:
        if k in args:
            opt[k] = args[args.index(k) + 1]
    splits = {"train": compare.parse_which(opt["--train"]), "heldout": compare.parse_which(opt["--hold"])}
    track = json.load(open(d / "track.json"))
    target = str(d / "target.wav")
    level = None

    lines = []
    for i, p in enumerate(track["phrases"]):
        ph = [x for w in p["words"] for x in w["phones"]]
        split = next((k for k, w in splits.items() if i in w), None)
        lines.append({"i": i, "text": p["text"], "t0": round(ph[0][1] - 0.15, 3), "t1": round(ph[-1][2] + 0.25, 3), "split": split})

    voices = {}
    for f in (REPO / "src" / "voices" / "local").glob("*.json"):
        v = json.load(open(f))
        if v.get("fit"):
            voices[str(pathlib.Path(v["fit"]).resolve())] = f.stem

    cands = []
    if (d / "fit" / "default-full.wav").exists():
        cands.append({"id": "default", "dir": d / "fit", "wav": d / "fit" / "default-full.wav", "label": "Untuned male", "about": "The engine's male body and source defaults, on the singer's timing, pitch and dynamics.", "params": None})
    for fd in sorted(d.glob("fit*")):
        best = fd / "best.json"
        if not best.exists():
            continue
        params = json.load(open(best))["params"]
        wav = fd / "best-full.wav"
        if not wav.exists():
            if level is None:
                level = [round(float(v), 2) for v in fit.level_curve(target, track)]
            fit.render(fit.track_for(track, params, None, level, track["duration"]), str(wav))
        about = (fd / "about.txt").read_text().strip().split("\n", 1) if (fd / "about.txt").exists() else [fd.name, ""]
        cands.append({"id": fd.name, "dir": fd, "wav": wav, "label": about[0], "about": about[1].strip() if len(about) > 1 else "", "params": params, "voice": voices.get(str(best.resolve()))})

    judge = None
    out = []
    for c in cands:
        cache_path = c["dir"] / ("catalog-cache-default.json" if c["id"] == "default" else "catalog-cache.json")
        cache = json.load(open(cache_path)) if cache_path.exists() else {}
        key = stamp(c["wav"]) + "|" + opt["--train"] + "|" + opt["--hold"]
        if cache.get("key") != key:
            if judge is None:
                from judge import Judge

                judge = Judge(d)
            cache = {"key": key, "scores": {}, "judges": {}}
            for name, which in splits.items():
                s, terms, groups = fit.score_file(target, str(c["wav"]), track, which)
                cache["scores"][name] = {"score": round(s, 4), "groups": {k: round(v, 3) for k, v in groups.items()}}
                cache["judges"][name] = judge.judge(str(c["wav"]), which)
            json.dump(cache, open(cache_path, "w"), indent=1)
            print(f"measured {c['id']}", flush=True)
        out.append({
            "id": c["id"],
            "label": c["label"],
            "about": c["about"],
            "wav": str(c["wav"].relative_to(d)),
            "voice": c.get("voice"),
            "params": c["params"],
            "scores": cache["scores"],
            "judges": cache["judges"],
        })
    cat = {"title": d.parent.name + " / " + d.name, "target": "target.wav", "lines": lines, "candidates": out}
    if (d / "accompaniment.wav").exists():
        cat["accompaniment"] = "accompaniment.wav"
    json.dump(cat, open(d / "candidates.json", "w"), indent=1)
    print(f"{d / 'candidates.json'}: {len(out)} candidates, {len(lines)} lines")


if __name__ == "__main__":
    main()
