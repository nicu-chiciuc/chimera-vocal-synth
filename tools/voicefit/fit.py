"""Fit the synth's voice to a recording: CMA-ES over body and source settings, each candidate
sung with `chimera render track` and scored with compare.py.

  fit.py <dir> [--train 0-3] [--hold 4-7] [--gens 40] [--pop 16] [--seed 1]
         [--from <dir>/fit/best.json] [--out fit] [--speaker]

<dir> holds target.wav and track.json (prepare.py). The fit sees only the training phrases; the
held-out phrases are scored at the end to show whether the voice carries over. Writes
<dir>/<out>/: log.jsonl (every candidate), best.json (the settings), best-full.wav (the whole
excerpt in the best voice), default-full.wav, and report.txt.
"""
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import cma
import librosa
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compare  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CHIMERA = REPO / "target" / "release" / "chimera"

# name, low, high, default (the engine's DEFAULT_SOURCE and the male body)
PARAMS = [
    ("scale", 0.92, 1.15, 1.0),
    ("pharynx", 0.85, 1.35, 1.0),
    ("pressure", 4.0, 18.0, 8.0),
    ("tension", 0.05, 0.95, 0.5),
    ("breathiness", 0.0, 0.9, 0.15),
    ("fry", 0.0, 0.7, 0.0),
    ("roughness", 0.0, 0.9, 0.1),
    ("brightness", 0.0, 0.9, 0.25),
    ("drift", 0.0, 20.0, 5.0),
    ("dyn", 0.0, 1.5, 0.0),
    # habitual articulation: offsets added to every planned target (cm, or 0..1 for lipWidth) ...
    ("off.larynx", -0.8, 0.6, 0.0),
    ("off.jaw", -0.25, 0.35, 0.0),
    ("off.lipProt", -0.2, 0.4, 0.0),
    ("off.root", -0.4, 0.4, 0.0),
    # ... and the rest position of what the plans leave alone (the engine's NEUTRAL by default)
    ("rest.epilarynx", 0.0, 0.8, 0.0),
    ("rest.ventricular", 0.0, 0.6, 0.0),
    ("rest.velum", -0.15, 0.3, -0.15),
    # lax articulation: tongue-body targets pulled this far toward neutral
    ("reduce", 0.0, 0.6, 0.0),
]


def decode(x):
    return {n: lo + (hi - lo) * float(np.clip(v, 0, 1)) for (n, lo, hi, _), v in zip(PARAMS, x)}


def encode(p):
    return [(p[n] - lo) / (hi - lo) for n, lo, hi, _ in PARAMS]


def level_curve(target, track):
    """The singer's level (dB re the median vowel level), 5 ms hop: vowel frames only, smoothed
    over 100 ms and held across consonants and pauses, so it carries syllable dynamics and not
    the consonants' own dips."""
    y, _ = librosa.load(target, sr=compare.FS, mono=True)
    hop = compare.HOP
    rms = librosa.feature.rms(y=y, frame_length=1440, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    vow = compare.frames_in([(a, b) for p, a, b in compare.phones(track) if compare.is_vowel(p)], len(db))
    idx = np.flatnonzero(vow)
    held = np.interp(np.arange(len(db)), idx, db[idx])
    k = 20
    sm = np.convolve(held, np.ones(k) / k, mode="same")
    return np.clip(sm - np.median(db[vow]), -15, 6)


def track_for(base, p, which, level, duration=None):
    t = dict(base)
    t["phrases"] = [ph for i, ph in enumerate(base["phrases"]) if which is None or i in which]
    if duration is None:
        duration = max(x[2] for ph in t["phrases"] for w in ph["words"] for x in w["phones"]) + 0.4
    t["duration"] = round(duration, 3)
    t["anatomy"] = {"scale": p["scale"], "pharynx": p["pharynx"]}
    t["source"] = {k: p[k] for k in ("tension", "breathiness", "fry", "roughness", "brightness", "drift", "pressure")}
    t["pressure"] = {"hop": compare.HOP / compare.FS, "values": [round(p["pressure"] * 10 ** (p["dyn"] * v / 20), 3) for v in level]}
    t["articulation"] = {
        "offset": {k[4:]: v for k, v in p.items() if k.startswith("off.")},
        "posture": {k[5:]: v for k, v in p.items() if k.startswith("rest.")},
        "reduce": p.get("reduce", 0.0),
    }
    return t


def render(track, out_wav):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(track, f)
    try:
        subprocess.run([str(CHIMERA), "render", "track", f.name, out_wav.removesuffix(".wav")], check=True, capture_output=True)
    finally:
        os.unlink(f.name)


_ctx = {}

# --speaker: a seventh group, how unlike the singer the ECAPA speaker embedding of the training
# phrases is. Round 3 of the Hurt fit won on the six acoustic groups and lost speaker similarity
# on held-out phrases (0.222 -> 0.119): without it the fit trades identity for score.
SPEAKER_TOP = 0.6  # similarity at which the cost is 0
SPEAKER_UNIT = 0.1


class Speaker:
    def __init__(self, target, track, which):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.enc = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb", savedir=str(pathlib.Path.home() / ".cache/chimera-target/models/ecapa"), run_opts={"device": "cpu"}
        )
        self.spans = []
        for i, p in enumerate(track["phrases"]):
            if which is None or i in which:
                ph = [x for w in p["words"] for x in w["phones"]]
                self.spans.append((ph[0][1] - 0.1, ph[-1][2] + 0.2))
        self.ref = self.embed(target)

    def embed(self, path):
        y = librosa.load(path, sr=16000)[0]
        with self.torch.no_grad():
            e = self.torch.stack([self.enc.encode_batch(self.torch.tensor(y[int(a * 16000) : int(b * 16000)])[None]).squeeze() for a, b in self.spans]).mean(0)
        return e / e.norm()

    def similarity(self, path):
        return float(self.ref @ self.embed(path))


def _init(target, base, which, level, speaker):
    _ctx.update(base=base, which=which, level=level, tgt=compare.Analysis(target, base, which, base["f0"]))
    _ctx["speaker"] = Speaker(target, base, which) if speaker else None


def _evaluate(x):
    p = decode(x)
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        render(track_for(_ctx["base"], p, _ctx["which"], _ctx["level"]), wav)
        syn = compare.Analysis(wav, _ctx["base"], _ctx["which"], _ctx["base"]["f0"])
        score, _, gcost = compare.compare(_ctx["tgt"], syn)
        if _ctx["speaker"]:
            sim = _ctx["speaker"].similarity(wav)
            gcost["speaker"] = max(0.0, (SPEAKER_TOP - sim) / SPEAKER_UNIT)
            score = float(np.mean(list(gcost.values())))
    finally:
        os.unlink(wav)
    return score, gcost


def score_file(target, wav, base, which):
    tgt = compare.Analysis(target, base, which, base["f0"])
    syn = compare.Analysis(wav, base, which, base["f0"])
    return compare.compare(tgt, syn)


def main():
    args = sys.argv[1:]
    d = pathlib.Path(args[0])
    opt = {"--train": "0-3", "--hold": "4-7", "--gens": "40", "--pop": "16", "--seed": "1", "--from": "", "--out": "fit"}
    for k in opt:
        if k in args:
            opt[k] = args[args.index(k) + 1]
    train, hold = compare.parse_which(opt["--train"]), compare.parse_which(opt["--hold"])
    out = d / opt["--out"]
    out.mkdir(exist_ok=True)
    base = json.load(open(d / "track.json"))
    target = str(d / "target.wav")
    level = [round(float(v), 2) for v in level_curve(target, base)]
    start = {n: dflt for n, _, _, dflt in PARAMS}
    sigma = 0.25
    if opt["--from"]:
        # continue from an earlier fit's settings (new parameters start at their defaults)
        start.update(json.load(open(opt["--from"]))["params"])
        sigma = 0.15
    x0 = encode(start)
    es = cma.CMAEvolutionStrategy(x0, sigma, {"bounds": [0, 1], "popsize": int(opt["--pop"]), "seed": int(opt["--seed"]), "verbose": -9})
    best = (np.inf, None, None)
    log = open(out / "log.jsonl", "w")
    t_start = time.time()
    speaker = "--speaker" in args
    with cf.ProcessPoolExecutor(os.cpu_count(), initializer=_init, initargs=(target, base, train, level, speaker)) as pool:
        s0, g0 = pool.submit(_evaluate, x0).result()
        print(f"default: {s0:.3f}  " + " ".join(f"{k} {v:.2f}" for k, v in g0.items()), flush=True)
        for gen in range(int(opt["--gens"])):
            xs = es.ask()
            res = list(pool.map(_evaluate, xs))
            es.tell(xs, [r[0] for r in res])
            for x, (s, g) in zip(xs, res):
                log.write(json.dumps({"gen": gen, "score": s, "groups": g, "params": decode(x)}) + "\n")
                if s < best[0]:
                    best = (s, decode(x), g)
            log.flush()
            print(f"gen {gen:2d} ({time.time() - t_start:4.0f} s): best {best[0]:.3f}  gen median {np.median([r[0] for r in res]):.3f}  "
                  + " ".join(f"{k} {v:.2f}" for k, v in best[2].items()), flush=True)
    s, p, _ = best
    json.dump({"score": s, "params": p}, open(out / "best.json", "w"), indent=1)
    full = base["duration"]
    render(track_for(base, p, None, level, full), str(out / "best-full.wav"))
    default = {n: dflt for n, _, _, dflt in PARAMS}
    render(track_for(base, default, None, level, full), str(out / "default-full.wav"))
    lines = [f"best settings: " + ", ".join(f"{k} {v:.3f}" for k, v in p.items())]
    for name, which in (("train", train), ("held-out", hold)):
        for label, wav in (("default", out / "default-full.wav"), ("fitted", out / "best-full.wav")):
            lines.append(f"\n{name}, {label}:\n" + compare.report(*score_file(target, str(wav), base, which)))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
