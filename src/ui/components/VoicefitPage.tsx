/**
 * The voice-fit playthrough (`?voicefit[=<set>]`, dev server only; the set defaults to `song`): a
 * recording's song sung by each fitted voice (tools/voicefit), with a 3D performer on a stage
 * singing along. A set is a folder under the dev server's voicefit folder with a candidates.json
 * (tools/voicefit/catalog.py); its recording and band stay on this machine.
 *
 * One player for the whole track: switching voice goes on from the same moment. Every voice is
 * level-matched to the singer, and the band (the separated instrumental) can play under it at the
 * record's own balance. The performer replays the playing voice's recorded telemetry, with the
 * sung words at its mouth, a hand strumming on the band's onsets and the other climbing the neck
 * with the pitch.
 *
 * Keys: space plays and pauses, left/right seek 5 s, 1-9 switch voice.
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chip } from "./Screen.tsx";
import type { HeadFeed } from "./AnatomyView3D.tsx";
import { TeleTrack } from "../../show/tele.ts";
import { getModel } from "../useModel.ts";
import { VoiceModel } from "../analysis.ts";
import type { BodyMix } from "../../wasm/types.ts";
import type { ViewFlags } from "../render/anatomy.ts";
import { mouthWords } from "../mouthWords.ts";
import { wordAt, type SungWords } from "../../show/words.ts";
import { V } from "../colors.ts";

// three.js loads with the stage
const AnatomyView3D = lazy(() => import("./AnatomyView3D.tsx").then((m) => ({ default: m.AnatomyView3D })));

interface Line {
  i: number;
  text: string;
  t0: number;
  t1: number;
  /** each word as aligned: text, start, end (s) */
  words?: [string, number, number][];
}

interface Candidate {
  id: string;
  label: string;
  about: string;
  wav: string;
  /** what the page plays: an MP3 of the render */
  audio?: string;
  /** the level that matches the voice to the singer over his sung lines */
  gain?: number;
  /** the voice's telemetry over the excerpt (render track --tele) */
  tele?: string;
  body?: BodyMix;
}

interface Catalog {
  title: string;
  target: string;
  /** the separated instrumental on the same timeline (prepare.py backing), and its MP3 */
  accompaniment?: string;
  band?: string;
  /** the track's length (s) */
  duration?: number;
  /** the band's strums (note onsets in the instrumental, s): the performer's right hand */
  strums?: number[];
  lines: Line[];
  candidates: Candidate[];
}

const ROOT = `${import.meta.env.BASE_URL}voicefit/`;
const BACKING = "backing";
/** the performer's look: airflow and sound pressure on, no measurement labels */
const VIEW: ViewFlags = { pressure: true, particles: true, grid: false, calipers: false };
const FORK = "https://github.com/nicu-chiciuc/chimera-vocal-synth";
const ORIGINAL = "https://github.com/jamesondh/chimera-vocal-synth";

function setName(): string {
  const s = new URLSearchParams(window.location.search).get("voicefit");
  return s && /^[\w-]+$/.test(s) ? s : "hurt";
}

/** the last index of a sorted list at or before t, or −1 */
function lastAtOrBefore(xs: number[], t: number): number {
  let lo = 0, hi = xs.length - 1, k = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= t) {
      k = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return k;
}

/** the right hand across the strings at t: each strum sweeps it, down and up in turn, in 110 ms */
function strumAt(on: number[], t: number): number {
  const k = lastAtOrBefore(on, t);
  if (k < 0) return 1;
  const to = k % 2 === 0 ? -1 : 1;
  const u = Math.min(1, (t - on[k]) / 0.11);
  return -to + 2 * to * u * u * (3 - 2 * u);
}

/** the beats the foot keeps: the band's onsets, no closer than 0.4 s apart */
function tapsOf(on: number[]): number[] {
  const out: number[] = [];
  for (const t of on) if (!out.length || t - out[out.length - 1] >= 0.4) out.push(t);
  return out;
}

/** the right toe at t: up before each beat, down on it (0..1) */
function tapAt(taps: number[], t: number): number {
  const k = lastAtOrBefore(taps, t) + 1;
  if (k >= taps.length) return 0;
  const left = taps[k] - t;
  return left < 0.3 ? Math.sin(Math.PI * (1 - left / 0.3)) : 0;
}

/** "Round 2: + articulation habits" → "Round 2" */
const shortLabel = (c: Candidate) => c.label.split(":")[0];
const clock = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export function VoicefitPage() {
  const set = setName();
  const base = `${ROOT}${set}/`;
  const [cat, setCat] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [music, setMusic] = useState(true);
  const [video, setVideo] = useState(true);
  const [sets, setSets] = useState<string[]>([]);
  /** voices still loading (the play button says so) */
  const [pending, setPending] = useState(0);
  const [teles, setTeles] = useState<Map<string, TeleTrack>>(new Map());
  /** the voice the player plays */
  const [sel, setSel] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  /** where the player stands while paused (s) */
  const [pos, setPos] = useState(0);
  const [, setTick] = useState(0);
  const ctx = useRef<AudioContext | null>(null);
  const buffers = useRef(new Map<string, AudioBuffer>());
  const loading = useRef(new Map<string, Promise<AudioBuffer | null>>());
  const current = useRef<AudioBufferSourceNode | null>(null);
  const musicSrc = useRef<AudioBufferSourceNode | null>(null);
  /** the playhead: started at audio time `at` from `t0` */
  const clockRef = useRef<{ at: number; t0: number } | null>(null);
  const epoch = useRef(0);

  useEffect(() => {
    fetch(ROOT + "sets.json", { cache: "no-store" })
      .then((r) => (r.ok ? (r.json() as Promise<string[]>) : []))
      .then(setSets)
      .catch(() => {});
    fetch(base + "candidates.json", { cache: "no-store" })
      .then(async (r) => {
        if (!r.ok) throw new Error(`${set}/candidates.json: ${r.status} ${await r.text()}`);
        const c = (await r.json()) as Catalog;
        setCat(c);
        setSel(c.candidates.find((x) => x.id !== "default")?.id ?? c.candidates[0]?.id ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [base, set]);

  const audio = useCallback(() => (ctx.current ??= new AudioContext()), []);

  /** a voice's (or the band's) audio, fetched and decoded once, when first wanted: a phone loads
   *  only what it plays (the MP3 when the catalog has one) */
  const load = useCallback(
    (id: string): Promise<AudioBuffer | null> => {
      const have = loading.current.get(id);
      if (have) return have;
      const c = cat?.candidates.find((x) => x.id === id);
      const file = id === BACKING ? (cat?.band ?? cat?.accompaniment) : (c?.audio ?? c?.wav);
      if (!file) return Promise.resolve(null);
      setPending((n) => n + 1);
      const p = fetch(base + file)
        .then(async (r) => {
          if (!r.ok) throw new Error(`${file}: ${r.status}`);
          const b = await audio().decodeAudioData(await r.arrayBuffer());
          buffers.current.set(id, b);
          return b;
        })
        .catch((e) => {
          setError(e instanceof Error ? e.message : String(e));
          return null;
        })
        .finally(() => setPending((n) => n - 1));
      loading.current.set(id, p);
      return p;
    },
    [cat, base, audio],
  );
  /** a voice's telemetry, for the performer, when it is first shown */
  const teleLoading = useRef(new Set<string>());
  const loadTele = useCallback(
    (id: string) => {
      const c = cat?.candidates.find((x) => x.id === id);
      if (!c?.tele || teleLoading.current.has(id)) return;
      teleLoading.current.add(id);
      void fetch(base + c.tele)
        .then(async (r) => (r.ok ? new TeleTrack(await r.arrayBuffer()) : null))
        .then((t) => t && setTeles((m) => new Map(m).set(id, t)));
    },
    [cat, base],
  );
  // the band and the first voice straight away
  useEffect(() => {
    if (!cat || !sel) return;
    void load(BACKING);
    void load(sel);
    loadTele(sel);
  }, [cat, sel, load, loadTele]);

  const ready = pending === 0 && !!sel && buffers.current.has(sel);
  const end = cat?.duration ?? buffers.current.get(BACKING)?.duration ?? 0;

  /** the playhead (s), or null while paused */
  const now = useCallback((): number | null => {
    const c = clockRef.current, ac = ctx.current;
    if (!c || !ac) return null;
    return c.t0 + Math.max(0, ac.currentTime - c.at);
  }, []);

  /** ends the track: the voice's or the band's end, whichever is later */
  const timer = useRef<number | null>(null);
  const halt = useCallback(() => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
    if (current.current) {
      current.current.onended = null;
      current.current.stop();
      current.current = null;
    }
    musicSrc.current?.stop();
    musicSrc.current = null;
    clockRef.current = null;
    setPlaying(false);
  }, []);

  /** play voice `id` from `from` to the end, the band with it if it is on */
  const start = useCallback(
    (id: string, from: number, withMusic: boolean) => {
      const b = buffers.current.get(id);
      if (!b || !end) return;
      halt();
      const t0 = Math.max(0, Math.min(end - 0.05, from));
      const ac = audio();
      void ac.resume();
      const src = ac.createBufferSource();
      src.buffer = b;
      const gain = ac.createGain();
      gain.gain.value = cat?.candidates.find((c) => c.id === id)?.gain ?? 1;
      src.connect(gain).connect(ac.destination);
      // voice and band start on the same sample of the same timeline
      const at = ac.currentTime + 0.03;
      const band = buffers.current.get(BACKING);
      if (withMusic && band && t0 < band.duration) {
        const m = ac.createBufferSource();
        m.buffer = band;
        m.connect(ac.destination);
        m.start(at, t0);
        musicSrc.current = m;
      }
      if (t0 < b.duration) src.start(at, t0);
      current.current = src;
      clockRef.current = { at, t0 };
      timer.current = window.setTimeout(() => {
        halt();
        setPos(0);
      }, (end - t0) * 1000 + 100);
      setSel(id);
      setPlaying(true);
      epoch.current++;
    },
    [audio, halt, end, cat],
  );

  const toggle = () => {
    if (!sel) return;
    const t = now();
    if (t !== null) {
      halt();
      setPos(t);
    } else start(sel, pos >= end - 0.1 ? 0 : pos, music);
  };
  const seek = (t: number) => {
    const to = Math.max(0, Math.min(end - 0.05, t));
    if (now() !== null && sel) start(sel, to, music);
    else setPos(to);
  };
  /** another voice from the same moment; one not loaded yet takes over once it is (the voice
   *  playing goes on meanwhile) */
  const switchTo = (id: string) => {
    setSel(id);
    loadTele(id);
    if (buffers.current.has(id)) {
      const t = now();
      if (t !== null) start(id, t, music);
      return;
    }
    void load(id).then(() => {
      const t = now();
      if (t !== null && controls.current.sel === id) start(id, t, controls.current.music);
    });
  };
  /** the band on or off, from the same moment */
  const setBand = (on: boolean) => {
    setMusic(on);
    const t = now();
    if (t !== null && sel) start(sel, t, on);
  };
  const controls = useRef({ toggle, seek, switchTo, now, pos, sel, music });
  controls.current = { toggle, seek, switchTo, now, pos, sel, music };

  // the time and the slider while playing
  useEffect(() => {
    const id = setInterval(() => clockRef.current && setTick((n) => n + 1), 100);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!cat || (e.target instanceof HTMLInputElement && e.target.type !== "range" && e.target.type !== "checkbox")) return;
      const c = controls.current;
      if (e.key === " ") {
        e.preventDefault();
        c.toggle();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        c.seek((c.now() ?? c.pos) + (e.key === "ArrowRight" ? 5 : -5));
      } else if (/^[1-9]$/.test(e.key) && cat.candidates[Number(e.key) - 1]) c.switchTo(cat.candidates[Number(e.key) - 1].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cat]);

  useEffect(
    () => () => {
      halt();
      void ctx.current?.close();
    },
    [halt],
  );

  // the words at the mouth, as the singer sang them (the aligner's word times)
  const words = useMemo<SungWords | null>(() => {
    const all = cat?.lines.flatMap((l) => l.words ?? []) ?? [];
    if (!all.length) return null;
    return { times: all.map(([, a, b]) => [a, b] as [number, number]), shown: all.map(([w]) => w), parent: all.map((_, i) => i) };
  }, [cat]);
  const overlay = useMemo(() => (words ? mouthWords(() => (clockRef.current ? wordAt(words, now() ?? -1) : null), V.green, 1) : undefined), [words, now]);

  // the performer: the selected voice's movements, reading the playhead each frame
  // the selected voice's movements once loaded; until then the last voice shown (dropping the view
  // while they load would remount it and replay its opening turn)
  const lastShown = useRef<string | null>(null);
  const performer = sel && teles.has(sel) ? sel : lastShown.current && teles.has(lastShown.current) ? lastShown.current : null;
  lastShown.current = performer;
  const byId = useMemo(() => new Map(cat?.candidates.map((c) => [c.id, c]) ?? []), [cat]);
  const feed = useMemo<HeadFeed | null>(() => {
    const c = performer ? byId.get(performer) : undefined;
    const track = performer ? teles.get(performer) : undefined;
    if (!c || !track) return null;
    const model = getModel(c.body ?? track.body, track.sampleRate);
    let last = VoiceModel.articFromArray(track.message(0).t.artic);
    const strums = music ? (cat?.strums ?? []) : [];
    const taps = tapsOf(strums);
    // the left hand climbs the neck with the sung pitch (eased, held through unvoiced sounds)
    let fret = 0.3;
    return {
      model,
      view: VIEW,
      stage: true,
      frame: () => {
        const t = now();
        if (t === null) return { tele: null, artic: last, epoch: epoch.current, pose: { strum: 1, fret, tap: 0, pluck: Infinity } };
        const msg = track.message(track.indexAt(t));
        last = VoiceModel.articFromArray(msg.t.artic);
        if (msg.t.f0 > 40 && msg.t.voicing > 0.3) fret += (Math.min(1, Math.max(0, Math.log2(msg.t.f0 / 85) / 1.1)) - fret) * 0.08;
        // the right hand strums on the band's onsets, the strings ring from the last one, the
        // toe keeps the beat
        const k = lastAtOrBefore(strums, t);
        const pose = { strum: strums.length ? strumAt(strums, t) : 1, fret, tap: tapAt(taps, t), pluck: k < 0 ? Infinity : t - strums[k] };
        return { tele: msg, artic: last, epoch: epoch.current, pose };
      },
    };
  }, [performer, byId, teles, now, cat, music]);

  if (error)
    return (
      <div className="vf">
        <p className="vf-msg">
          {error}. Build the catalog with <code>tools/voicefit/catalog.py &lt;dir&gt;</code> and point the dev server at the folder that holds it (<code>VOICEFIT_DIR</code>, or a link at <code>out/voicefit</code>).
        </p>
      </div>
    );
  if (!cat) return <div className="vf" />;
  const at = now() ?? pos;
  const line = cat.lines[Math.max(0, lastAtOrBefore(cat.lines.map((l) => l.t0), at))];
  const lyric = line && at >= line.t0 && at <= line.t1 + 1.5 ? line.text : "";

  return (
    <div className="vf">
      <header className="vf-head">
        <h1 className="vf-title">{cat.title}</h1>
        <span className="vf-sub">sung by CHIMERA, a simulated vocal tract</span>
        <nav className="vf-links">
          {sets
            .filter((s) => s !== set)
            .map((s) => (
              <a key={s} href={`?voicefit=${s}`}>
                {s.replace(/-/g, " ").replace(/^\w/, (x) => x.toUpperCase())} →
              </a>
            ))}
          <a href={FORK} target="_blank" rel="noreferrer">
            This fork
          </a>
          <a href={ORIGINAL} target="_blank" rel="noreferrer">
            CHIMERA by jamesondh
          </a>
        </nav>
      </header>

      {video && (
        <section className="vf-stage" aria-label="The performer, in 3D">
          {feed ? (
            <Suspense fallback={null}>
              {/* one view for every voice: a new one would replay the opening camera turn; a voice
                  with another build keeps the camera where it is, centred on the new head */}
              <AnatomyView3D feed={feed} readouts={false} overlay={overlay} />
            </Suspense>
          ) : (
            <p className="vf-msg">Loading the performer…</p>
          )}
        </section>
      )}

      <section className="vf-player" aria-label="Player">
        <div className="vf-plays" role="radiogroup" aria-label="Voice">
          {cat.candidates.map((c) => (
            <Chip key={c.id} on={sel === c.id} onClick={() => switchTo(c.id)} title={c.about}>
              {shortLabel(c)}
            </Chip>
          ))}
        </div>
        <div className="vf-bar">
          <Chip toggle={false} className="vf-playbtn" onClick={toggle} disabled={!ready && !playing}>
            {playing ? "❚❚ Pause" : ready ? "▶ Play" : "Loading…"}
          </Chip>
          <span className="vf-time">
            {clock(at)} / {clock(end)}
          </span>
          <input
            type="range"
            aria-label="Position in the song"
            min={0}
            max={end || 1}
            step={0.1}
            value={Math.min(at, end || 1)}
            disabled={!ready}
            onChange={(e) => seek(Number(e.target.value))}
            style={{ "--pos": `${end ? (100 * at) / end : 0}%` } as React.CSSProperties}
          />
        </div>
        <div className="vf-opts">
          {cat.accompaniment && (
            <label className="vf-check">
              <input type="checkbox" checked={music} onChange={(e) => setBand(e.target.checked)} />
              Band
            </label>
          )}
          <label className="vf-check">
            <input type="checkbox" checked={video} onChange={(e) => setVideo(e.target.checked)} />
            Video
          </label>
          <span className="vf-keys">space play/pause · ← → 5 s · 1-{cat.candidates.length} voice</span>
        </div>
        <p className="vf-now">{lyric || " "}</p>
      </section>
    </div>
  );
}
