"""How well a rendition keeps time with the singer: sync.py <dir> <render.wav> [track.json]

For each phrase of <dir>/track.json, the lag (10 ms steps, up to ±600 ms) at which the
render's level envelope best matches the singer's (target.wav), and the correlation there.
A rendition that sings every word where the singer does scores about 0.9 on a dry solo voice
(Hurt); lines at 0.2-0.5 have words in the wrong place. The lag says which way: negative, the
render is late.
"""
import json
import pathlib
import sys

import librosa
import numpy as np

SR = 16000
HOP = 160  # 10 ms


def envelope(path, n=None):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if n is not None:
        y = y[:n]
    return y, np.log(librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0] + 1e-4)


def best_lag(x, y, max_lag=60):
    x = x - x.mean()
    y = y - y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y) + 1e-9
    best = (0, -1.0)
    for lag in range(-max_lag, max_lag + 1):
        c = (np.dot(x[lag:], y[: len(y) - lag]) if lag >= 0 else np.dot(x[:lag], y[-lag:])) / norm
        if c > best[1]:
            best = (lag, float(c))
    return best


def main(d, render, track_path=None):
    d = pathlib.Path(d)
    yt, et = envelope(d / "target.wav")
    _, es = envelope(render, len(yt))
    n = min(len(et), len(es))
    et, es = et[:n], es[:n]
    track = json.load(open(track_path or d / "track.json"))
    rows = []
    for i, p in enumerate(track["phrases"]):
        ph = [x for w in p["words"] for x in w["phones"]]
        if not ph:
            continue
        a, b = ph[0][1], ph[-1][2]
        fa, fb = max(0, int(a * SR / HOP) - 50), int(b * SR / HOP) + 50
        lag, c = best_lag(et[fa:fb], es[fa:fb])
        rows.append((lag * 10, c))
        print(f"{i:3d} lag {lag * 10:+5d} ms  corr {c:4.2f}  {p['text'][:50]}")
    print(f"median corr {np.median([c for _, c in rows]):.2f}, lines under 0.6: {sum(c < 0.6 for _, c in rows)} of {len(rows)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:4])
