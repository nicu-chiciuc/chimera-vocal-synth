"""Export a synthetic performance, its self-contained render input, and a local playback set.

  export.py <source-dir> <candidate-id> <destination>

The author needs the local fit data when exporting. Recipients need only the Rust CLI to
regenerate the exported score. No reference recording or backing audio is copied.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

import fit


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("candidate")
    parser.add_argument("destination", type=pathlib.Path)
    args = parser.parse_args()
    source, dest = args.source.resolve(), args.destination.resolve()
    catalog = json.loads((source / "candidates.json").read_text())
    candidate = next((c for c in catalog["candidates"] if c["id"] == args.candidate), None)
    if candidate is None:
        parser.error(f"candidate {args.candidate!r} not found")
    dest.mkdir(parents=True, exist_ok=True)
    saved = (source / candidate["wav"]).parent / "render-track.json"
    if saved.exists():
        score = json.loads(saved.read_text())
    else:
        base = json.loads((source / "track.json").read_text())
        params = {name: default for name, lo, hi, default in fit.PARAMS}
        params.update(candidate.get("params") or {})
        level = [round(float(v), 2) for v in fit.level_curve(str(source / "target.wav"), base)]
        score = fit.track_for(base, params, None, level, base["duration"])
    # Export the engine's input, not creator-side paths, caches, or recording metadata.
    score = {k: score[k] for k in ("body", "duration", "source", "posture", "habits", "phrases", "f0", "pressure", "sustain_vowels") if k in score}
    write_json(dest / "score.json", score)
    write_json(dest / "voice.json", {k: score[k] for k in ("body", "source", "posture", "habits") if k in score})
    fit.render(score, str(dest / "vocals.wav"), str(dest / "animation.tele"))
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(dest / "vocals.wav"),
                    "-codec:a", "libmp3lame", "-b:a", "192k", str(dest / "vocals.mp3")], check=True)
    lines = []
    for i, phrase in enumerate(score["phrases"]):
        phones = [p for w in phrase["words"] for p in w["phones"]]
        if not phones:
            continue
        words = [[w["w"], w["phones"][0][1], w["phones"][-1][2]] for w in phrase["words"] if w["phones"]]
        lines.append({"i": i, "text": phrase.get("text", " ".join(w[0] for w in words)),
                      "t0": max(0, phones[0][1] - 0.15), "t1": phones[-1][2] + 0.25, "words": words})
    write_json(dest / "candidates.json", {
        "title": catalog["title"], "target": "", "duration": score["duration"], "lines": lines,
        "candidates": [{"id": args.candidate, "label": candidate["label"],
                        "about": "Synthetic vocals without accompaniment", "wav": "vocals.mp3",
                        "audio": "vocals.mp3", "tele": "animation.tele", "gain": 1,
                        "body": score.get("body", {"female": 0, "child": 0})}],
    })
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=fit.REPO, text=True).strip()
    names = ["score.json", "voice.json", "candidates.json", "vocals.mp3", "animation.tele"]
    write_json(dest / "manifest.json", {
        "title": catalog["title"], "candidate": args.candidate, "engine_revision": revision,
        "duration_seconds": score["duration"], "original_recording_included": False,
        "backing_audio_included": False,
        "sha256": {n: hashlib.sha256((dest / n).read_bytes()).hexdigest() for n in names},
        "wav_master": "vocals.wav is generated locally and included in the download bundle; regenerate it from score.json when cloning git",
    })
    print(f"{catalog['title']}: {dest}, {score['duration']:.2f} s, voice {candidate['label']}", flush=True)


if __name__ == "__main__":
    main()
