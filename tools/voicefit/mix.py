"""A voice over the band for the whole excerpt, as one file to listen to anywhere:

  mix.py <dir> <out-dir> [candidate ...]

For each candidate (a folder of <dir> with best-full.wav: default, fit, fit2, ...; all of them if
none given), writes <out-dir>/<candidate>.mp3: the render level-matched to the singer over his
sung lines (as the comparison page plays it) over <dir>/accompaniment.wav, the record's own
balance, peak-normalised to -1 dBFS. The band is the recording's: keep the files to yourself.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf

FS = 48000


def main():
    d, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[3:] or sorted(p.parent.name for p in d.glob("*/best-full.wav"))
    track = json.load(open(d / "track.json"))
    target = librosa.load(d / "target.wav", sr=FS)[0]
    band = librosa.load(d / "accompaniment.wav", sr=FS)[0]
    n = max(len(target), len(band))
    sung = np.zeros(n, bool)
    for p in track["phrases"]:
        ph = [x for w in p["words"] for x in w["phones"]]
        sung[int(ph[0][1] * FS) : int(ph[-1][2] * FS)] = True
    rms = lambda x: float(np.sqrt(np.mean(x[sung[: len(x)]] ** 2)) + 1e-9)  # noqa: E731
    ref = rms(target)
    for name in names:
        voice = librosa.load(d / name / "best-full.wav", sr=FS)[0]
        mix = np.zeros(n, np.float32)
        mix[: len(band)] += band
        mix[: len(voice)] += voice * (ref / rms(voice))
        mix *= 10 ** (-1 / 20) / np.abs(mix).max()
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, mix, FS)
            dest = out / f"{name}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name, "-b:a", "192k", str(dest)], check=True)
        print(f"{dest}: {n / FS:.0f} s")


if __name__ == "__main__":
    main()
