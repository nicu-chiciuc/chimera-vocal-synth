"""Listening file: each phrase as the singer sang it, then as each render sings it.

  ab.py <dir> <out.wav> <render.wav>... [--phrases 0-7]

Renders are level-matched to the singer per phrase (RMS), with 0.4 s of silence between parts.
The output holds the singer's audio: keep it next to the recording, out of git.
"""
import json
import pathlib
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compare  # noqa: E402

FS = compare.FS


def main():
    args = sys.argv[1:]
    which = None
    if "--phrases" in args:
        i = args.index("--phrases")
        which = compare.parse_which(args[i + 1])
        del args[i : i + 2]
    d = pathlib.Path(args[0])
    track = json.load(open(d / "track.json"))
    target = librosa.load(d / "target.wav", sr=FS)[0]
    renders = [librosa.load(w, sr=FS)[0] for w in args[2:]]
    gap = np.zeros(int(0.4 * FS), np.float32)
    out = []
    for i, p in enumerate(track["phrases"]):
        if which is not None and i not in which:
            continue
        ph = [x for w in p["words"] for x in w["phones"]]
        a, b = int((ph[0][1] - 0.15) * FS), int((ph[-1][2] + 0.25) * FS)
        t = target[a:b]
        rms = np.sqrt(np.mean(t**2)) + 1e-9
        out += [t, gap]
        for r in renders:
            x = r[a:b]
            out += [x * rms / (np.sqrt(np.mean(x**2)) + 1e-9), gap]
        out.append(gap)
    y = np.concatenate(out)
    sf.write(args[1], 0.9 * y / np.abs(y).max(), FS)
    print(f"{args[1]}: {len(y) / FS:.1f} s")


if __name__ == "__main__":
    main()
