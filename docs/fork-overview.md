# What this fork contains

CHIMERA is a voice synthesizer with a browser interface. Its Rust engine models vocal folds,
the vocal tract, and airflow to generate sound. The same engine runs in the browser through
WebAssembly and on your computer through a command-line renderer.

This fork adds a way to reproduce the phrasing of a recorded singer, compare voices, repair
some timing errors, and replay the result with an animated performer. It also includes two
exported performances that you can regenerate from their scores.

## How a song becomes a synthetic performance

The recording tools first identify the words, the sounds within each word, the singer's pitch,
and changes in loudness. They write those controls into a `score.json` file. A score is the
performance recipe: which sounds to make, when to make them, and how to drive the voice.

The Rust renderer reads the score and generates a new waveform. It can also record the
mouth, airway, and vocal-fold states into `animation.tele`. The browser player uses that
animation alongside the audio, so the performer follows the rendered voice.

Whisper, wav2vec2, and Montreal Forced Aligner help transcribe and align the reference.
Speaker embeddings and speech recognition help compare candidates. Those models analyze
recordings and results; the singing waveform comes from the physical synthesizer.

The exported scores use JSON rather than MIDI. They include individual speech-sound timings,
a continuous pitch curve, breath pressure, and voice settings. The app also has live MIDI
input, but these recorded-song performances are driven by the exported controls.

## Where the code lives

All paths below are relative to your `chimera-vocal-synth` checkout.

| Folder | Purpose |
| --- | --- |
| `crates/chimera/` | Rust voice model, articulation, and acoustic simulation. |
| `crates/chimera-cli/` | Offline renderer. `render track` reads the recorded-performance score. |
| `crates/chimera-wasm/` | Connects the Rust engine to the browser. |
| `src/ui/` | React interface and Three.js performer. |
| `src/worklet/` | Browser audio processing. |
| `tools/voicefit/` | Python alignment, fitting, timing repairs, comparisons, and export. |
| `performances/` | Shared scores, voice settings, animation, playback metadata, and audio outputs. |
| `docs/` | Explanations, setup instructions, and experiment results. |

Rust is the sound engine. React and TypeScript are the interface. Three.js draws the stage.
Bun runs the development commands, and Vite serves the app. Python is needed for preparing
and fitting new references, but not for rendering an already exported score. There is no
hosted backend in this workflow.

## How a performance is stored

The two shared sets are `performances/hurt/` and `performances/country-roads/`.

| File | What it stores |
| --- | --- |
| `score.json` | The full performance recipe, including timing, pitch, dynamics, and voice controls. |
| `voice.json` | The voice's body, source, posture, and articulation settings. |
| `animation.tele` | Mouth and airway states recorded while that score was rendered. |
| `candidates.json` | The title, timed words, audio path, and animation path used by the player. |
| `vocals.mp3` | A rendered listening copy. The score can generate a WAV instead. |
| `manifest.json` | The code revision used to render the files and their checksums. |

The MP3 is an output, not the synthesizer or the project itself. You can share the repository
and point readers at the scores, renderer, and player. See [the performance guide](../performances/README.md)
for commands that regenerate WAV audio and animation without the original recording.

## What stays on the creator's computer

The working recordings and experiments are kept outside the repository, under dataset folders
such as `~/datasets/hurt/song`, `~/datasets/country/song`, and `~/datasets/born/song`.
They contain reference vocals, backing tracks, aligner output, candidate renders, and audits.
`out/voicefit` can link those folders into the local development player.

`out/`, compiled Rust files under `target/`, the generated WebAssembly file, local WAV masters,
and the full local profile library under `src/voices/local/` are ignored by git. The selected
voice settings and self-contained scores under `performances/` are committed separately.
Desktop screen recordings and trimmed videos are separate files, not part of the app.

## What this branch adds to main

The branch adds recorded-performance rendering and voice-profile controls, the local comparison
player and animated stage, reference preparation and fitting tools, reviewed timing repairs,
and portable performance exports. The upstream project supplied the underlying synthesizer.

Timing and pronunciation remain imperfect. Country Roads includes nine accepted word repairs,
but its audit still flags unresolved short vowels. The separate long-diphthong prototype is
disabled by default because it did not pass the clarity check. Backing harmonies have not
been added.

For the complete workflow, see [render and compare a recorded performance](voicefit.md).
For the underlying sound model, see [the acoustic architecture](architecture.md).
