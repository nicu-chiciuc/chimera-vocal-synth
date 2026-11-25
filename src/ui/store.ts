import { create } from "zustand";
import { engine } from "../audio/AudioEngine.ts";
import { inPlugin, setParam } from "../audio/pluginBridge.ts";
import { DEFAULT_SOURCE, type VoiceSource } from "../wasm/engine.ts";
import { NEUTRAL, restTip, type HumanArtic } from "../wasm/engine.ts";
import { defaultF0, PRESETS, sameBody, type BodyMix } from "./body.ts";
import { creatureById } from "../wasm/engine.ts";
import type { ParamDef } from "../wasm/types.ts";
import { BEAT_PRESETS, BASS_NOTE, beatReady, parseBeat, type BeatStep } from "../wasm/engine.ts";
import { alloc, setVoiceCount, showFirstVoice, showNewestVoice, singNoteOn, singNoteOff, stopSinging } from "./singing.ts";
import { MAX_VOICES } from "./voiceAlloc.ts";
import { liveVoices } from "./voices.ts";
import { stopSpeaking } from "./speech.ts";
import { connectMidi, disconnectMidi, type MidiStatus } from "./midi.ts";

export type Mode = "play" | "speak" | "sing" | "bestiary" | "beatbox";

export interface BeatState {
  steps: (BeatStep | null)[];
  bpm: number;
  swing: number;
  playing: boolean;
  /** preset the pattern came from (null once edited) */
  preset: string | null;
  /** in the plugin: the sequencer follows the host's transport (the setting), it is doing so now, and the host's tempo */
  follow: boolean;
  following: boolean;
  hostBpm: number;
}

export interface ViewOptions {
  pressure: boolean;
  particles: boolean;
  grid: boolean;
  calipers: boolean;
  /** the face screen shows the head in 3D (turn it with the pointer; no handles) */
  threeD: boolean;
}

interface State {
  started: boolean;
  starting: boolean;
  startError: string | null;
  /** the device couldn't keep up at 48 kHz: the voice runs at 32 kHz (see AudioEngine) */
  lightAudio: boolean;
  /** the device can't keep up even so: the voice breaks up or goes silent */
  overloaded: boolean;
  /** the body shown; while the body pad is dragged it runs ahead of the voice (see previewBody) */
  body: BodyMix;
  source: VoiceSource;
  artic: HumanArtic;
  tipFree: boolean;
  volume: number;
  /** sustained breath (drone) independent of keys */
  hold: boolean;
  heldNotes: number[];
  /** the notes a chord's voices sound now, for the keys and the Keys header (in the plugin the host's notes too) */
  voiceNotes: number[];
  /** how many voices Sing and Play have: a chord's notes, one voice each (1: one voice, legato) */
  voices: number;
  setVoices: (n: number) => void;
  view: ViewOptions;
  mode: Mode;
  setMode: (m: Mode) => void;
  /** active non-human creature in the bestiary */
  creature: string;
  cparams: Record<string, Record<string, number>>;
  setCreature: (id: string) => void;
  setCParam: (key: string, v: number) => void;
  playCall: (id: string, pitch?: number) => void;
  /** what the keys do in the bestiary: hold the voice at a pitch (null) or trigger this call */
  keyCall: string | null;
  setKeyCall: (id: string | null) => void;
  beat: BeatState;
  /** close-mic sound (proximity bass + compression) in beatbox mode */
  mic: boolean;
  /** hum note for new hum steps: follows the last key played */
  bassNote: number;
  setBeatSteps: (steps: (BeatStep | null)[], preset?: string | null) => void;
  /** one sound per step: set it, or clear the step if that sound is already there */
  toggleBeatStep: (i: number, id: string) => void;
  setBeatTempo: (v: { bpm?: number; swing?: number }) => void;
  setBeatPlaying: (on: boolean) => void;
  setBeatFollow: (on: boolean) => void;
  loadBeatPreset: (name: string) => void;
  beatHit: (id: string) => void;
  setMic: (on: boolean) => void;
  start: () => Promise<void>;
  /** switch the voice to this body (rebuilds its airway) */
  setBody: (b: BodyMix) => void;
  /** show this body without rebuilding the voice yet (a drag in progress; setBody commits it) */
  previewBody: (b: BodyMix) => void;
  setSource: (v: Partial<VoiceSource>) => void;
  setArtic: (v: Partial<HumanArtic>, tipFree?: boolean) => void;
  setVolume: (v: number) => void;
  setHold: (h: boolean) => void;
  /** Breathe: on (starting the audio first, if need be) or off */
  toggleBreath: () => Promise<void>;
  /** a key went down (velocity 1..127; the on-screen keys send 96) */
  noteOn: (midi: number, velocity?: number) => void;
  noteOff: (midi: number) => void;
  midi: { status: MidiStatus; inputs: string[] };
  setMidi: (on: boolean) => void;
  setView: (v: Partial<ViewOptions>) => void;
}

const midiToHz = (m: number) => 440 * Math.pow(2, (m - 69) / 12);

/**
 * Map a key to a creature's pitch range: shift the whole keyboard by whole octaves so middle C
 * lands near the middle of the range (adjacent octaves stay distinct), then fold stragglers in.
 */
export function creatureKeyHz(midi: number, p: Pick<ParamDef, "min" | "max">): number {
  const shift = Math.round(Math.log2(Math.sqrt(p.min * p.max) / midiToHz(60)));
  let f = midiToHz(midi) * Math.pow(2, shift);
  while (f > p.max) f /= 2;
  while (f < p.min) f *= 2;
  return Math.min(p.max, f);
}

/** The most voices the web app offers: each costs about 15 % of a laptop's core (docs/plugin.md, "The web app's chords"). */
export const WEB_MAX_VOICES = 4;

/**
 * The voices the web app starts with: one, unless the build sets VITE_START_VOICES. The preview's
 * build sets 3 (.github/workflows/preview.yml), so a visitor's first two keys are a chord with no
 * chip to find. In the plugin the host's parameter sets the count.
 */
const START_VOICES = inPlugin ? 1 : Math.max(1, Math.min(WEB_MAX_VOICES, Math.round(Number(import.meta.env.VITE_START_VOICES) || 1)));

/** The voices that play: a chord's in Sing and Play, one in the other tabs. */
const playingVoices = (s: { mode: Mode; voices: number }) => (s.mode === "sing" || s.mode === "play" ? s.voices : 1);

/** Play's chord: each voice's note for the pitch lines, the chord for the Keys header, the head on the newest. */
function followPlayVoices(set: (p: Partial<State>) => void): void {
  const notes = alloc.keys.slice(0, alloc.count).map((k) => (k.length ? k[k.length - 1] : -1));
  liveVoices.notes = alloc.count > 1 ? notes : [];
  set({ voiceNotes: alloc.count > 1 ? notes.filter((n) => n >= 0) : [] });
  showNewestVoice();
}

/**
 * The Pitch dial moved while a key had this voice of a Play chord: the voice takes the dial's
 * pitch when its last key lifts, as in the plugin (Hold then breathes there).
 */
const dialF0: (number | undefined)[] = [];

/** Play's keys let go, on every voice of a chord (a tab switch, a new count of voices, a new body, a rebuilt engine). */
function releasePlayVoices(set: (p: Partial<State>) => void): void {
  for (let v = 1; v < alloc.count; v++) engine.gate(0, v);
  alloc.clear();
  dialF0.length = 0;
  showFirstVoice();
  liveVoices.notes = [];
  set({ voiceNotes: [] });
}

function updateGate(s: { hold: boolean; heldNotes: number[]; mode: Mode; keyCall: string | null }) {
  if (inPlugin) {
    // the keys' breath is Rust's, from whichever keys are down
    engine.pluginHold(s.hold);
    return;
  }
  if (alloc.count > 1 && s.mode === "play") {
    // each voice of a chord breathes while its key is down; Hold keeps the first one breathing,
    // and with no key down that is the voice the head shows
    for (let v = 0; v < alloc.count; v++) engine.gate((v === 0 && s.hold) || alloc.busy(v) ? 1 : 0, v);
    if (s.hold && alloc.newest() === null) showFirstVoice();
    return;
  }
  // keys that trigger calls leave the breath to the call; sung notes breathe by themselves
  const keysBreathe = !(s.mode === "bestiary" && s.keyCall) && s.mode !== "sing";
  engine.gate(s.hold || (keysBreathe && s.heldNotes.length > 0) ? 1 : 0);
}

/** Everything the voice plays from, sent to an engine that knows none of it (a new one). */
function sendState(s: State) {
  engine.setVolume(s.volume);
  if (!sameBody(s.body, PRESETS.male)) engine.setBody(s.body);
  engine.setSource(s.source);
  engine.setArtic(s.artic, s.tipFree);
  if (s.mode === "bestiary") engine.setCreature(s.creature, s.cparams[s.creature]);
  if (s.mode === "beatbox") {
    engine.setMic(s.mic);
    engine.setBeat({ steps: s.beat.steps, bpm: s.beat.bpm, swing: s.beat.swing, playing: s.beat.playing });
  }
  setVoiceCount(playingVoices(s));
  // a chord whose keys went down before the engine existed: each voice's own pitch, which the
  // shared one above (the newest key's) has just written over
  if (alloc.count > 1 && s.mode === "play") {
    for (let v = 0; v < alloc.count; v++) {
      const keys = alloc.keys[v];
      if (keys.length) engine.voiceSource(v, { f0: midiToHz(keys[keys.length - 1]) }, true);
    }
  }
  updateGate(s);
}

let startPromise: Promise<void> | null = null;

/**
 * A key pressed before the audio exists is sent once it does: only if that same press is still
 * down (not a later press of the same key) and the mode and body have not changed since.
 */
let pressSerial = 0;
const pressOf = new Map<number, number>();
let session = 0;

/**
 * Per-tab voice settings, kept out of the other tabs. Play's Whisper is the one switch for
 * voicing, and no other tab shows it: left on, Speak and Sing went voiceless with no way to tell
 * why. Beatbox closes the jaw and lips to its ready pose, and the pose stayed after leaving.
 */
let whisperInPlay = false;
let poseBeforeBeat: { artic: HumanArtic; tipFree: boolean } | null = null;

const FIRST_BEAT = BEAT_PRESETS[0];

export const useStore = create<State>((set, get) => ({
  started: false,
  starting: false,
  startError: null,
  lightAudio: false,
  overloaded: false,
  body: PRESETS.male,
  source: { ...DEFAULT_SOURCE },
  artic: (() => {
    const tip = restTip(NEUTRAL.jaw, NEUTRAL.tbX, NEUTRAL.tbY);
    return { ...NEUTRAL, ttX: tip.x, ttY: tip.y };
  })(),
  tipFree: true,
  volume: 0.8,
  hold: false,
  heldNotes: [],
  voiceNotes: [],
  voices: START_VOICES,
  setVoices: (count) => {
    const n = Math.max(1, Math.min(inPlugin ? MAX_VOICES : WEB_MAX_VOICES, Math.round(count)));
    if (inPlugin) {
      // the host's parameter: Rust deals the notes (docs/plugin.md, "Voices")
      set({ voices: n });
      setParam("voices", n);
      return;
    }
    const s = get();
    if (n === s.voices) return;
    // notes held now end: the keys are dealt to the voices again from the next press (and a
    // press still waiting for the audio to start is not sent when it does)
    if (s.heldNotes.length) {
      pressOf.clear();
      if (s.mode === "sing") stopSinging();
      if (s.mode === "play") releasePlayVoices(set);
      if (s.mode === "sing" || s.mode === "play") set({ heldNotes: [] });
    }
    set({ voices: n });
    // before the audio starts this goes nowhere, and the start sends it
    setVoiceCount(playingVoices(get()));
    updateGate(get());
  },
  view: { pressure: true, particles: true, grid: false, calipers: true, threeD: false },
  // the app opens on Sing: hearing a song is the quickest way in (the plugin opens there too)
  mode: "sing",
  setMode: (m) => {
    const prev = get().mode;
    if (prev === m) return;
    if (inPlugin) {
      // Rust stops the notes and the breath when the mode changes, and sets Beatbox's posture
      set((s) => ({ mode: m, heldNotes: [], hold: false, artic: m === "beatbox" ? { ...s.artic, ...beatReady(s.body) } : s.artic }));
      engine.pluginMode(m);
      return;
    }
    session++;
    // notes held across the switch would keep breathing into the other voice
    set({ mode: m, heldNotes: [] });
    if (prev === "play" && !get().source.voiced) {
      whisperInPlay = true;
      get().setSource({ voiced: true });
    }
    if (m === "play" && whisperInPlay) {
      whisperInPlay = false;
      get().setSource({ voiced: false });
    }
    // speech belongs to Speak: under a creature the human clock stops and it would resume later
    if (prev === "speak" || m === "bestiary") stopSpeaking();
    if ((prev === "bestiary") !== (m === "bestiary")) {
      // a sustain belongs to the voice it was started on
      set({ hold: false });
      const s = get();
      engine.setCreature(m === "bestiary" ? s.creature : "human", s.cparams[s.creature]);
    }
    if (prev === "beatbox") {
      get().setBeatPlaying(false);
      engine.setMic(false);
      if (poseBeforeBeat) get().setArtic(poseBeforeBeat.artic, poseBeforeBeat.tipFree);
      poseBeforeBeat = null;
    }
    if (prev === "sing") stopSinging();
    if (prev === "play") releasePlayVoices(set);
    // a chord's voices are Sing's and Play's: the other tabs have one
    if (playingVoices(get()) !== alloc.count) setVoiceCount(playingVoices(get()));
    // speech and song breathe by themselves: a held breath would drone under them
    if (m === "speak" || m === "sing") set({ hold: false });
    if (m === "beatbox") {
      // a beatboxer waits with the jaw nearly closed; the sounds bring their own breath
      set({ hold: false });
      poseBeforeBeat = { artic: get().artic, tipFree: get().tipFree };
      get().setArtic(beatReady(get().body));
      engine.setMic(get().mic);
    }
    updateGate(get());
  },
  creature: "songbird",
  cparams: {},
  keyCall: null,
  setKeyCall: (id) => {
    set({ keyCall: id, heldNotes: [] });
    updateGate(get());
  },
  beat: { steps: parseBeat(FIRST_BEAT.text), bpm: FIRST_BEAT.bpm, swing: FIRST_BEAT.swing, playing: false, preset: FIRST_BEAT.name, follow: true, following: false, hostBpm: 0 },
  mic: true,
  bassNote: BASS_NOTE,
  setBeatSteps: (steps, preset = null) => {
    set((s) => ({ beat: { ...s.beat, steps, preset } }));
    engine.setBeat({ steps, preset });
  },
  toggleBeatStep: (i, id) => {
    const steps = get().beat.steps.slice();
    steps[i] = steps[i]?.id === id ? null : id === "m" ? { id, note: get().bassNote } : { id };
    get().setBeatSteps(steps);
  },
  setBeatTempo: (v) => {
    set((s) => ({ beat: { ...s.beat, ...v } }));
    engine.setBeat(v);
  },
  setBeatPlaying: (on) => {
    if (on === get().beat.playing) return;
    set((s) => ({ beat: { ...s.beat, playing: on } }));
    if (inPlugin) {
      engine.setBeat({ playing: on });
      return;
    }
    if (!on) {
      engine.setBeat({ playing: false });
      return;
    }
    void get()
      .start()
      .then(() => {
        const b = get().beat;
        // the audio didn't start: the beat isn't playing, so its button offers Play (a retry), not Stop
        if (!get().started) set((s) => ({ beat: { ...s.beat, playing: false } }));
        else if (b.playing) engine.setBeat({ steps: b.steps, bpm: b.bpm, swing: b.swing, playing: true });
      });
  },
  setBeatFollow: (on) => {
    set((s) => ({ beat: { ...s.beat, follow: on } }));
    engine.pluginFollowHost(on);
  },
  loadBeatPreset: (name) => {
    const p = BEAT_PRESETS.find((x) => x.name === name);
    if (!p) return;
    get().setBeatSteps(parseBeat(p.text), p.name);
    get().setBeatTempo({ bpm: p.bpm, swing: p.swing });
  },
  beatHit: (id) => {
    void get()
      .start()
      .then(() => {
        if (get().started) engine.beatHit(id, id === "m" ? get().bassNote : undefined);
      });
  },
  setMic: (on) => {
    set({ mic: on });
    if (inPlugin || get().mode === "beatbox") engine.setMic(on);
  },
  setCreature: (id) => {
    if (id === get().creature) return;
    set({ creature: id, keyCall: null, heldNotes: [] });
    if (get().mode === "bestiary") engine.setCreature(id, get().cparams[id]);
    updateGate(get());
  },
  setCParam: (key, v) => {
    const id = get().creature;
    set((s) => ({ cparams: { ...s.cparams, [id]: { ...(s.cparams[id] ?? {}), [key]: v } } }));
    engine.setCreatureParams({ [key]: v });
  },
  playCall: (id, pitch) => {
    void get()
      .start()
      .then(() => {
        if (get().started) engine.playCall(id, pitch);
      });
  },
  start: () => {
    if (get().started) return Promise.resolve();
    if (startPromise) return startPromise;
    set({ starting: true, startError: null });
    startPromise = (async () => {
      try {
        await engine.start();
        // anything set before audio existed was not delivered: send the whole state now
        sendState(get());
        set({ started: true, starting: false });
      } catch (e) {
        set({ starting: false, startError: e instanceof Error ? e.message : String(e) });
      } finally {
        startPromise = null;
      }
    })();
    return startPromise;
  },
  setBody: (b) => {
    session++;
    // the new voice starts without the old one's score: a sung note would hang
    if (get().mode === "sing") stopSinging();
    const f0 = defaultF0(b);
    // nor does speech carry over (and the new voice's clock starts again from zero)
    if (get().mode === "speak") stopSpeaking();
    // a Play chord ends, as Sing's notes do and as in the plugin: the new body starts every voice
    // at its own pitch, which would leave the chord's keys lit over a unison
    const chord = get().mode === "play" && alloc.count > 1;
    if (chord) {
      releasePlayVoices(set);
      set({ heldNotes: [] });
    }
    set((s) => ({ body: b, source: { ...s.source, f0 } }));
    engine.setBody(b);
    if (chord) updateGate(get());
  },
  previewBody: (b) => set({ body: b }),
  setSource: (v) => {
    set((s) => ({ source: { ...s.source, ...v } }));
    if (!inPlugin && v.f0 !== undefined && alloc.count > 1 && get().mode === "play" && alloc.newest() !== null) {
      // a chord's keys keep their pitches: the Pitch dial moves only the voices with no key down
      const { f0, ...rest } = v;
      if (Object.keys(rest).length) engine.setSource(rest);
      for (let k = 0; k < alloc.count; k++) {
        if (alloc.busy(k)) dialF0[k] = f0;
        else engine.voiceSource(k, { f0 });
      }
      return;
    }
    engine.setSource(v);
  },
  setArtic: (v, tipFree) => {
    set((s) => ({ artic: { ...s.artic, ...v }, tipFree: tipFree ?? s.tipFree }));
    engine.setArtic(v, tipFree);
  },
  setVolume: (v) => {
    set({ volume: v });
    engine.setVolume(v);
  },
  setHold: (h) => {
    set({ hold: h });
    updateGate(get());
  },
  toggleBreath: async () => {
    if (get().hold) return get().setHold(false);
    // on at once, so a second press while the audio starts turns it off again; switching tabs
    // meanwhile clears it as any switch does (setMode)
    get().setHold(true);
    await get().start();
    // the audio didn't start: nothing breathes, so the button offers Breathe (a retry) again
    if (!get().started) get().setHold(false);
  },
  noteOn: (midi, velocity = 96) => {
    const held = get().heldNotes.filter((n) => n !== midi).concat(midi);
    set({ heldNotes: held });
    if (inPlugin) {
      // new hum steps take the last key played
      if (get().mode === "beatbox") set({ bassNote: midi });
      engine.pluginNote(midi, true, velocity);
      return;
    }
    const s = get();
    const press = ++pressSerial;
    pressOf.set(midi, press);
    // a sung note or a call is a one-off message, lost if sent before the audio exists (held
    // pitches and breath are resent at startup): send it once audio is up, if still pressed
    if (!s.started && (s.mode === "sing" || (s.mode === "bestiary" && s.keyCall))) {
      const sess = session;
      void s.start().then(() => {
        const t = get();
        if (t.started && session === sess && pressOf.get(midi) === press) t.noteOn(midi, velocity);
      });
      return;
    }
    if (s.mode === "sing") singNoteOn(midi, velocity, s.body);
    else if (s.mode === "bestiary") {
      const pp = creatureById(s.creature)?.params.find((p) => p.role === "pitch");
      const f = pp ? creatureKeyHz(midi, pp) : undefined;
      if (s.keyCall) s.playCall(s.keyCall, f);
      else if (pp && f) s.setCParam(pp.key, f);
    } else if (s.mode === "beatbox") {
      // keys hum a bass note under the beat (lips shut, nose open); new hum steps take it too
      set({ bassNote: midi });
      if (held.length === 1) s.setArtic({ lipAp: -0.25, velum: 0.9 });
      s.setSource({ f0: midiToHz(midi) });
    } else if (alloc.count > 1) {
      // a chord: the key's own voice (src/ui/voiceAlloc.ts), from silence at its pitch with no glide
      const v = alloc.pick(midi);
      const fromSilence = !alloc.busy(v) && !(v === 0 && s.hold);
      alloc.press(midi, v);
      dialF0[v] = undefined;
      engine.voiceSource(v, { f0: midiToHz(midi) }, fromSilence);
      // the Pitch dial shows the newest note
      set((st) => ({ source: { ...st.source, f0: midiToHz(midi) } }));
      followPlayVoices(set);
    } else s.setSource({ f0: midiToHz(midi) });
    updateGate(get());
  },
  noteOff: (midi) => {
    pressOf.delete(midi);
    const held = get().heldNotes.filter((n) => n !== midi);
    set({ heldNotes: held });
    if (inPlugin) {
      engine.pluginNote(midi, false);
      return;
    }
    const s = get();
    if (s.mode === "sing") {
      singNoteOff(midi, s.body);
      return;
    }
    if (s.mode === "beatbox" && !held.length) s.setArtic({ lipAp: beatReady(s.body).lipAp, velum: NEUTRAL.velum });
    if (s.mode === "play" && alloc.count > 1) {
      // a chord: the key's voice goes back to its older key still down, or falls silent
      const lifted = alloc.lift(midi);
      if (lifted && lifted.back !== null) engine.voiceSource(lifted.voice, { f0: midiToHz(lifted.back) });
      else if (lifted && dialF0[lifted.voice] !== undefined) {
        engine.voiceSource(lifted.voice, { f0: dialF0[lifted.voice] });
        dialF0[lifted.voice] = undefined;
      }
      // the Pitch dial and the Keys header show the newest note still down (the dial, turned
      // since, shows what it was turned to)
      const newest = alloc.newest();
      if (newest !== null && !dialF0.some((f) => f !== undefined)) set((st) => ({ source: { ...st.source, f0: midiToHz(alloc.keys[newest].at(-1)!) } }));
      followPlayVoices(set);
      updateGate(get());
      return;
    }
    if (held.length) {
      // fall back to the most recent note still held (legato)
      const last = held[held.length - 1];
      if (s.mode !== "bestiary") s.setSource({ f0: midiToHz(last) });
      else if (!s.keyCall) {
        const pp = creatureById(s.creature)?.params.find((p) => p.role === "pitch");
        if (pp) s.setCParam(pp.key, creatureKeyHz(last, pp));
      }
    }
    updateGate(get());
  },
  setView: (v) => set((s) => ({ view: { ...s.view, ...v } })),
  midi: { status: "off", inputs: [] },
  setMidi: (on) => {
    const noteOff = (n: number) => get().noteOff(n);
    if (!on) {
      disconnectMidi();
      set({ midi: { status: "off", inputs: [] } });
      return;
    }
    set({ midi: { status: "connecting", inputs: [] } });
    // the click that connects is also the gesture that may start audio
    void get().start();
    connectMidi({
      noteOn: (n, v) => {
        void get().start();
        get().noteOn(n, v);
      },
      noteOff,
      inputs: (names) => set({ midi: { status: "on", inputs: names } }),
    }).catch(() => {
      if (get().midi.status === "connecting") set({ midi: { status: "denied", inputs: [] } });
    });
  },
}));

// the voices the page starts with are the allocator's too, from the first moment. Nothing shows
// without this today (Sing's first key waits for the start, which sets the count, and so does a
// change of tab); a page that opened on Play would give its first keys to voice 0, which then
// breathes on no voice.
alloc.count = playingVoices(useStore.getState());

engine.onStatus(() => useStore.setState({ lightAudio: engine.light, overloaded: engine.overloaded }));
// the lighter audio is a new engine: the phrase being sung or spoken is gone (the score was the
// old engine's), and the rest is sent again
engine.onRebuilt(() => {
  session++;
  stopSinging();
  stopSpeaking();
  // (a Play chord's keys too: they would stay lit, and named in the Keys header, over silence)
  releasePlayVoices((p) => useStore.setState(p));
  useStore.setState({ heldNotes: [] });
  sendState(useStore.getState());
});
