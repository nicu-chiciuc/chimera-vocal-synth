# Render and compare a recorded performance

This fork adds a recording-to-synth workflow. It takes aligned words and the singer's pitch
curve, then generates new audio with CHIMERA's physical voice model. The default male voice
does not require a fitted profile or a reference recording to speak or sing your own score.

The browser app, CLI, stage renderer, alignment tools, fitting tools, and timing repairs are
in git. Reference recordings, separated backing tracks, cover-song scores, fitted artist
profiles, generated covers, and local comparison sets are excluded. A fresh clone therefore
does not include the Country Roads, Hurt, or Born to Run listening sessions.

## Hear the default male voice

Follow [the app setup](../README.md#run-it). With Rust installed, generate the included original
vowel exercise from the repository root:

```sh
cargo run --release -p chimera-cli -- render notes examples/male-voice.json out/male-voice male
ffmpeg -i out/male-voice.wav -codec:a libmp3lame -b:a 160k out/male-voice.mp3
```

The generated WAV contains only the synthetic voice. The repository includes an MP3 at
[`public/demos/male-voice.mp3`](../public/demos/male-voice.mp3). No recording separation,
Python environment, or model downloads are needed for this example. FFmpeg is needed only
for the MP3 conversion. The note numbers are MIDI pitches and the second value is duration
in beats. Supply one lyric syllable per sounding note; a nonpositive pitch is a rest.

For an original spoken example:

```sh
cargo run --release -p chimera-cli -- render text out/spoken "This voice comes from a simulated airway." male
```

## Set up the optional recording tools

The recording workflow uses Python because the existing acoustic-analysis and alignment tools
use Librosa, Praat, PyTorch, and Montreal Forced Aligner. The browser app does not need them.
Use Python 3.12 and FFmpeg. Create an environment outside the repository:

```sh
python3.12 -m venv "$HOME/.cache/chimera-target/.venv"
source "$HOME/.cache/chimera-target/.venv/bin/activate"
python -m pip install -r tools/voicefit/requirements.txt
cargo build --release -p chimera-cli
```

The requirements list records the direct packages and model-loading versions used on Apple Silicon for this fork.
It is not a complete environment lock. First use of transcription, alignment, or the independent
judges downloads their model weights. Fitting with both judges can be slow and uses multiple
CPU workers; a better acoustic score alone does not establish a better voice.

Install micromamba for your platform, then create a separate Montreal Forced Aligner environment.
These commands assume a macOS or Linux shell:

```sh
micromamba create -y -p "$HOME/.cache/chimera-mfa/env" -c conda-forge montreal-forced-aligner=3.4.3 python=3.11
export MFA_ROOT_DIR="$HOME/.cache/chimera-mfa/mfa"
micromamba run -p "$HOME/.cache/chimera-mfa/env" mfa model download acoustic english_us_arpa --version v3.0.0
micromamba run -p "$HOME/.cache/chimera-mfa/env" mfa model download dictionary english_us_arpa --version v3.0.0
```

Keep the `MFA_ROOT_DIR` setting in the shell used for alignment; `prepare.py` expects the
dictionary at that cache location. The existing `tools/tsp/setup.sh` automates the pinned MFA
setup on Linux. It downloads a Linux binary, so do not use it to install MFA on macOS.

## Prepare material you can use

Use your own isolated vocal, or material you have permission to process. If you separate a mix
elsewhere, use its isolated vocal as the input here. These tools do not bundle a separator or
retrieve commercial songs. Keep all source and output files under a dataset directory outside git.

From the repository root, with the analysis environment activated:

```sh
mkdir -p "$HOME/datasets/my-voice"
python tools/voicefit/transcribe.py /path/to/vocal.wav "$HOME/datasets/my-voice/words.json"
python tools/voicefit/prepare.py cut /path/to/vocal.wav "$HOME/datasets/my-voice/words.json" 0 30 "$HOME/datasets/my-voice/song"
```

The example cuts the first 30 seconds. Review the transcription and phrase text before alignment.
For lyrics you can use, `lyrics.py words.json lyrics.txt corrected-words.json` maps supplied words
onto the transcription's timing; use that corrected JSON in `prepare.py cut` instead.

Set a short directory variable for the remaining commands:

```sh
voicefit_song="$HOME/datasets/my-voice/song"
python tools/voicefit/align.py "$voicefit_song"
micromamba run -p "$HOME/.cache/chimera-mfa/env" mfa align --clean --output_format json "$voicefit_song/mfa/corpus" english_us_arpa english_us_arpa "$voicefit_song/mfa/aligned"
python tools/voicefit/prepare.py collect "$voicefit_song"
python tools/voicefit/refine.py "$voicefit_song" --dry
python tools/voicefit/refine.py "$voicefit_song"
```

`micromamba run` uses the separate MFA environment without changing the analysis environment.
`align.py` rewrites the generated MFA corpus, so preserve
manual corpus edits before running it again. The collected `track.json` contains the aligned
phones and pitch curve; review its timing before fitting a voice.

## Render, check, and listen

```sh
target/release/chimera render track "$voicefit_song/track.json" "$voicefit_song/untuned" --tele "$voicefit_song/untuned.tele"
python tools/voicefit/sync.py "$voicefit_song" "$voicefit_song/untuned.wav"
python tools/voicefit/catalog.py "$voicefit_song" --title "My voice" --train 0-3 --hold 4-7 --no-measure
mkdir -p out/voicefit
ln -s "$voicefit_song" out/voicefit/my-voice
bun run dev
```

Open `http://127.0.0.1:5317/?voicefit=my-voice`. Use a fresh set name if that link already exists.
The comparison route serves local datasets only during development; `bun run build` does not
include them. The app and its CLI require no hosted backend.

An optional backing track can be prepared with `prepare.py backing /path/to/backing.wav
"$voicefit_song"` before building the catalog. Without it, playback is voice-only.

Choose train and hold ranges that exist in your excerpt, with different lines in each. For fitting:

```sh
python tools/voicefit/fit.py "$voicefit_song" --train 0-3 --hold 4-7 --gens 10 --pop 8 --speaker --clarity
python tools/voicefit/judge.py "$voicefit_song" "$voicefit_song/fit/best-full.wav" --phrases 4-7
```

Compare against an untuned render on the same lines. Keep the fitting objective and its held-out
check separate. Voice similarity, word recognition, synchronization, and listening can disagree.
The current Country Roads work found the untuned male clearer than the fitted artist profiles.

## Audit short vowels

```sh
python tools/voicefit/refine.py "$voicefit_song" --onsets-only --dry
python tools/voicefit/refine.py "$voicefit_song" --short-vowels
```

The first command checks clipped phrase openings. Remove `--dry` to apply supported onset repairs.
The second audits every vowel and writes `vowel-audit.json` with proposed word-level repairs;
it does not change `track.json`. A duration at or below 80 ms is a review flag, not proof of error.

Render proposed changes separately and compare clarity and synchronization. Mark only passing
entries `"accepted": true` in the report, then apply them:

```sh
python tools/voicefit/refine.py "$voicefit_song" --apply-audit "$voicefit_song/vowel-audit.json"
python -m unittest discover -s tools/voicefit
```

Applying a reviewed report preserves word boundaries, rejects stale input, and is safe to retry.
Regenerate any affected audio, MP3, telemetry, catalog gains, and cached scores afterward; the
repair command changes the score, not previously rendered media. Long-diphthong timing is a
separate opt-in `"sustain_vowels": true` track setting. Its current listening prototype did not
pass the clarity guard and remains disabled by default.

## Share code and recordings separately

The code is [MIT licensed](../LICENSE); keep the original copyright notice. The bundled dictionary
and fonts have their own notices. Generated sound does not by itself clear rights in an existing
song's lyrics or melody. An a cappella cover still uses the song, even if it contains no samples
of the original performance. See the [Copyright Office's distinction between songs and recordings](https://www.copyright.gov/engage/musicians/).

For a public demo, use your own material, material cleared for your intended use, or an appropriate
public-domain work. Reproduction instructions do not confer permission to redistribute a song or
its reference audio. The included vowel exercise is independent of the local commercial-song tests.
