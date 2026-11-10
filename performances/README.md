# Synthetic vocal performances

These are CHIMERA's generated vocals without backing tracks:

- [Hurt, fitted voice round 3](https://raw.githubusercontent.com/nicu-chiciuc/chimera-vocal-synth/t3/fit-target-voice/performances/hurt/vocals.mp3)
- [Take Me Home, Country Roads, untuned male with timing repairs](https://raw.githubusercontent.com/nicu-chiciuc/chimera-vocal-synth/t3/fit-target-voice/performances/country-roads/vocals.mp3)

[Download both performances, WAV masters, scores, and animation](https://github.com/nicu-chiciuc/chimera-vocal-synth/releases/download/performances-2026-11-10/chimera-performances.zip).

The files keep the full performance timeline, including pauses and the instrumental introduction's
silence. They contain no original recording or backing audio. These are synthetic covers of
existing songs; the software's MIT license does not grant rights in their compositions.

Each folder contains:

- `vocals.mp3`: the finished vocal stem.
- `score.json`: aligned phonemes, pitch and pressure curves, and voice settings. It can render
  without the reference recording, a Python environment, or a fitted-profile installation.
- `voice.json`: the voice's body, source, posture, and articulation settings.
- `animation.tele`: the matching mouth and airway animation.
- `candidates.json`: a voice-only playback set for this app.
- `manifest.json`: the rendering code revision and checksums of the exported files.

WAV masters are generated locally and included in the download bundle, but excluded from git.
You can regenerate them with the commands below. Country Roads has nine accepted word-timing
repairs. Hurt includes four timing repairs in "tears", "wear", "someone", and "would";
some pronunciation problems remain. The opt-in long-vowel experiment is not used here.

## Regenerate the vocals

From the repository root, with Rust installed:

```sh
cargo build --release -p chimera-cli
mkdir -p out
target/release/chimera render track performances/hurt/score.json out/hurt-vocals --tele out/hurt.tele
target/release/chimera render track performances/country-roads/score.json out/country-roads-vocals --tele out/country-roads.tele
```

The commands write WAV files and animation data. To create an MP3 with FFmpeg:

```sh
ffmpeg -i out/hurt-vocals.wav -codec:a libmp3lame -b:a 192k out/hurt-vocals.mp3
```

The scores contain all timing and dynamic controls needed by the renderer. Rebuilding the
alignment or fitting a different voice is a separate process described in [the workflow](../docs/voicefit.md).

## Play with the animated performer

Follow [the app setup](../README.md#run-it), then start the dev server with these exported sets:

```sh
VOICEFIT_DIR="$PWD/performances" bun run dev
```

Open `http://127.0.0.1:5317/?voicefit=hurt` or
`http://127.0.0.1:5317/?voicefit=country-roads`. The page plays only the exported voice and uses
its matching animation. If you already have a dev server running, restart it with this setting.

To export another local candidate with the analysis environment installed:

```sh
python tools/voicefit/export.py /path/to/local/song candidate-id /path/to/export-folder
```

The export command renders new audio from the score. It does not copy the reference recording
or the band's track. Regenerating an exported score requires only the Rust renderer.
