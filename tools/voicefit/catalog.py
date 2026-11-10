"""Collect a fit's candidates for the comparison page (the app's `?voicefit`):

  catalog.py <dir> [--train 0-3] [--hold 4-7] [--title "Song name"] [--no-measure]

--no-measure skips the acoustic scores and the judges (the playthrough page doesn't show them).
Each voice and the band also get an MP3 (what the page plays), and each voice its gain: the
level that matches it to the singer over his sung lines.

Writes <dir>/candidates.json: the singer's lines (text, times, each word's times, training or
held-out), the band's strums (onsets in accompaniment.wav, if there is one) and every
candidate voice: the untuned default (<dir>/default/) and each <dir>/fit*/ with a best.json, each
sung over the whole excerpt (best-full.wav, and best-full.tele for the page's 3D head; rendered
if missing). A fit's best.json can be copied into another excerpt's folder (the whole song) to
hear the same voice there. For each: the settings, the body, the acoustic score per split
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
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compare  # noqa: E402
import fit  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


def stamp(p):
    s = os.stat(p)
    return f"{s.st_size}:{int(s.st_mtime)}"


def strums(path):
    """The band's strums: note onsets in the instrumental (librosa), for the performer's right hand."""
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    return [round(float(t), 3) for t in librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)]


def mp3(wav):
    """A 160 kbit/s MP3 beside a WAV (for the page: a phone fetches 5 MB a voice, not 25), kept
    while it is newer than the WAV."""
    out = wav.with_suffix(".mp3")
    if not out.exists() or out.stat().st_mtime < wav.stat().st_mtime:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "160k", str(out)], check=True)
    return out


def sung_rms(path, lines):
    """RMS over the singer's sung lines (a voice is matched to him, not to the silences)"""
    import librosa
    import numpy as np

    y, sr = librosa.load(path, sr=16000, mono=True)
    parts = [y[int(l["t0"] * sr) : int(l["t1"] * sr)] for l in lines]
    x = np.concatenate([p for p in parts if len(p)])
    return float(np.sqrt(np.mean(x**2)) + 1e-9)


def main():
    args = sys.argv[1:]
    d = pathlib.Path(args[0]).resolve()
    opt = {"--train": "0-3", "--hold": "4-7", "--title": ""}
    for k in opt:
        if k in args:
            opt[k] = args[args.index(k) + 1]
    measure = "--no-measure" not in args
    splits = {"train": compare.parse_which(opt["--train"]), "heldout": compare.parse_which(opt["--hold"])}
    track = json.load(open(d / "track.json"))
    target = str(d / "target.wav")
    level = None

    lines = []
    for i, p in enumerate(track["phrases"]):
        ph = [x for w in p["words"] for x in w["phones"]]
        split = next((k for k, w in splits.items() if i in w), None)
        words = [[w["w"], w["phones"][0][1], w["phones"][-1][2]] for w in p["words"] if w["phones"]]
        lines.append({"i": i, "text": p["text"], "t0": round(ph[0][1] - 0.15, 3), "t1": round(ph[-1][2] + 0.25, 3), "split": split, "words": words})

    # the app's profiles, by the settings they were made from (a fit's best.json may be copied to
    # another excerpt's folder: the settings are what identify it)
    key_of = lambda p: json.dumps(p, sort_keys=True)  # noqa: E731
    voices = {}
    for f in (REPO / "src" / "voices" / "local").glob("*.json"):
        v = json.load(open(f))
        if v.get("fit") and os.path.exists(v["fit"]):
            voices[key_of(json.load(open(v["fit"]))["params"])] = f.stem

    defaults = {n: dflt for n, _, _, dflt in fit.PARAMS}
    found = [(d / "default", defaults, ["Untuned male", "The engine's male body and source defaults, on the singer's timing, pitch and dynamics."])]
    for fd in sorted(d.glob("fit*")):
        if (fd / "best.json").exists():
            about = (fd / "about.txt").read_text().strip().split("\n", 1) if (fd / "about.txt").exists() else [fd.name, ""]
            found.append((fd, json.load(open(fd / "best.json"))["params"], about))
    cands = []
    for fd, params, about in found:
        wav, tele = fd / "best-full.wav", fd / "best-full.tele"
        if not wav.exists() or not tele.exists():
            if level is None:
                level = [round(float(v), 2) for v in fit.level_curve(target, track)]
            fd.mkdir(exist_ok=True)
            fit.render(fit.track_for(track, params, None, level, track["duration"]), str(wav), tele)
            print(f"rendered {fd.name}", flush=True)
        cands.append({
            "id": "default" if fd.name == "default" else fd.name,
            "dir": fd,
            "wav": wav,
            "tele": tele,
            "label": about[0],
            "about": about[1].strip() if len(about) > 1 else "",
            "params": None if params is defaults else params,
            "body": {"female": 0, "child": 0, "size": params["scale"], "pharynx": params["pharynx"]},
            "voice": voices.get(key_of(params)),
        })

    judge = None
    out = []
    ref = sung_rms(target, lines)
    for c in cands:
        cache_path = c["dir"] / "catalog-cache.json"
        cache = json.load(open(cache_path)) if cache_path.exists() else {}
        key = stamp(c["wav"]) + "|" + opt["--train"] + "|" + opt["--hold"]
        if not measure and cache.get("key") != key:
            cache = {"scores": {}, "judges": {}}
        elif cache.get("key") != key:
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
            "audio": str(mp3(c["wav"]).relative_to(d)),
            "gain": round(ref / sung_rms(str(c["wav"]), lines), 4),
            "tele": str(c["tele"].relative_to(d)),
            "body": c["body"],
            "voice": c.get("voice"),
            "params": c["params"],
            "scores": cache["scores"],
            "judges": cache["judges"],
        })
    cat = {"title": opt["--title"] or d.parent.name.replace("-", " ").capitalize(), "target": "target.wav", "lines": lines, "candidates": out}
    if (d / "accompaniment.wav").exists():
        cat["accompaniment"] = "accompaniment.wav"
        cat["band"] = str(mp3(d / "accompaniment.wav").relative_to(d))
        cat["duration"] = round(track["duration"], 3)
        cat["strums"] = strums(d / "accompaniment.wav")
    json.dump(cat, open(d / "candidates.json", "w"), indent=1)
    print(f"{d / 'candidates.json'}: {len(out)} candidates, {len(lines)} lines")


if __name__ == "__main__":
    main()
