/**
 * The voice-fit comparison page (`?voicefit=<set>`, dev server only): every candidate voice from
 * tools/voicefit against the recording it copies, the whole excerpt or line by line, with a 3D
 * performer on a stage singing whatever plays. A set is a folder under the dev server's voicefit
 * folder (`song`, `verse1`, ...) with a candidates.json (tools/voicefit/catalog.py).
 *
 * Blind by default: candidates get letters in an order shuffled once per browser, and their names
 * and measures stay hidden until Reveal. Every play is level-matched to the singer over the same
 * stretch, and by default the band (the separated instrumental) plays under it at the record's
 * own balance. The performer replays the playing voice's recorded telemetry; while the singer
 * plays, it moves as the voice it last showed (every voice sings the same phones on the same
 * timeline, so it keeps in time with him).
 *
 * Keys: up/down pick a line, 0 or C plays the singer, 1-9 the candidates, space the line through
 * all of them, Esc stops.
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chip, Screen } from "./Screen.tsx";
import type { HeadFeed } from "./AnatomyView3D.tsx";
import { voiceById } from "../../voices/index.ts";
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

type Split = "train" | "heldout";

interface Line {
  i: number;
  text: string;
  t0: number;
  t1: number;
  split: Split | null;
  /** each word as aligned: text, start, end (s) */
  words?: [string, number, number][];
}

interface Judged {
  similarity: number;
  wer: number;
  heard: string[];
}

interface Candidate {
  id: string;
  label: string;
  about: string;
  wav: string;
  /** the voice's telemetry over the excerpt (render track --tele) */
  tele?: string;
  body?: BodyMix;
  voice?: string | null;
  params: Record<string, number> | null;
  scores: Record<Split, { score: number; groups: Record<string, number> }>;
  judges: Record<Split, Judged>;
}

interface Catalog {
  title: string;
  target: string;
  /** the separated instrumental on the same timeline (prepare.py backing) */
  accompaniment?: string;
  /** the band's strums (note onsets in the instrumental, s): the performer's right hand */
  strums?: number[];
  lines: Line[];
  candidates: Candidate[];
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

/** a stretch of the timeline: one line, or the whole excerpt */
interface Span {
  key: string;
  t0: number;
  t1: number;
}

const ROOT = `${import.meta.env.BASE_URL}voicefit/`;
const TARGET = "target";
const BACKING = "backing";
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
/** the performer's look: airflow and sound pressure on, no measurement labels */
const VIEW: ViewFlags = { pressure: true, particles: true, grid: false, calipers: false };

function setName(): string {
  const s = new URLSearchParams(window.location.search).get("voicefit");
  return s && /^[\w-]+$/.test(s) ? s : "song";
}

/** A shuffle that stays the same in this browser (so letters keep their meaning across reloads). */
function blindOrder(ids: string[]): string[] {
  let seed = 0;
  try {
    seed = Number(localStorage.getItem("voicefit-seed")) || 0;
    if (!seed) {
      seed = 1 + Math.floor(Math.random() * 1e9);
      localStorage.setItem("voicefit-seed", String(seed));
    }
  } catch {
    seed = 12345;
  }
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  const out = [...ids].sort();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function rms(b: AudioBuffer, t0: number, t1: number): number {
  const x = b.getChannelData(0);
  const a = Math.max(0, Math.floor(t0 * b.sampleRate)), z = Math.min(x.length, Math.floor(t1 * b.sampleRate));
  let s = 0;
  for (let i = a; i < z; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, z - a)) + 1e-9;
}

const pct = (x: number) => `${Math.round(x * 100)} %`;
const clock = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export function VoicefitPage() {
  const set = setName();
  const base = `${ROOT}${set}/`;
  const [sets, setSets] = useState<string[]>([]);
  const [cat, setCat] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [blind, setBlind] = useState(true);
  // the band under every rendition, at the song's own balance: how a voice sits in the record
  const [music, setMusic] = useState(true);
  const [line, setLine] = useState(0);
  /** what is playing: voice id and span key */
  const [playing, setPlaying] = useState<{ id: string; span: string } | null>(null);
  const [loaded, setLoaded] = useState(0);
  const [teles, setTeles] = useState<Map<string, TeleTrack>>(new Map());
  /** the voice whose movements the performer shows */
  const [shown, setShown] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const ctx = useRef<AudioContext | null>(null);
  const buffers = useRef(new Map<string, AudioBuffer>());
  const current = useRef<AudioBufferSourceNode | null>(null);
  const musicSrc = useRef<AudioBufferSourceNode | null>(null);
  const queue = useRef<number | null>(null);
  /** where the playing span is on the timeline: started at audio time `at`, from t0 to t1 */
  const clockRef = useRef<{ at: number; t0: number; t1: number } | null>(null);
  const epoch = useRef(0);

  useEffect(() => {
    fetch(ROOT + "sets.json", { cache: "no-store" })
      .then((r) => (r.ok ? (r.json() as Promise<string[]>) : []))
      .then(setSets)
      .catch(() => {});
    fetch(base + "candidates.json", { cache: "no-store" })
      .then(async (r) => {
        if (!r.ok) throw new Error(`${set}/candidates.json: ${r.status} ${await r.text()}`);
        setCat((await r.json()) as Catalog);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [base, set]);

  const audio = useCallback(() => (ctx.current ??= new AudioContext()), []);

  // decode the recording, the band and every render once; load each voice's telemetry
  useEffect(() => {
    if (!cat) return;
    const files: [string, string][] = [[TARGET, cat.target], ...cat.candidates.map((c) => [c.id, c.wav] as [string, string])];
    if (cat.accompaniment) files.push([BACKING, cat.accompaniment]);
    let alive = true;
    void Promise.all(
      files.map(async ([id, file]) => {
        const r = await fetch(base + file);
        if (!r.ok) throw new Error(`${file}: ${r.status}`);
        const b = await audio().decodeAudioData(await r.arrayBuffer());
        if (!alive) return;
        buffers.current.set(id, b);
        setLoaded((n) => n + 1);
      }),
    ).catch((e) => setError(e instanceof Error ? e.message : String(e)));
    void Promise.all(
      cat.candidates
        .filter((c) => c.tele)
        .map(async (c) => {
          const r = await fetch(base + c.tele);
          return r.ok ? ([c.id, new TeleTrack(await r.arrayBuffer())] as const) : null;
        }),
    ).then((all) => alive && setTeles(new Map(all.filter((x) => x !== null))));
    return () => {
      alive = false;
    };
  }, [cat, audio, base]);

  const order = useMemo(() => (cat ? (blind ? blindOrder(cat.candidates.map((c) => c.id)) : cat.candidates.map((c) => c.id)) : []), [cat, blind]);
  const byId = useMemo(() => new Map(cat?.candidates.map((c) => [c.id, c]) ?? []), [cat]);
  const spans = useMemo<Span[]>(() => cat?.lines.map((l) => ({ key: `line${l.i}`, t0: l.t0, t1: l.t1 })) ?? [], [cat]);
  // the whole recording, the band's intro and outro included (its length once it is decoded)
  const whole = useMemo<Span | null>(() => {
    if (!cat || !cat.lines.length) return null;
    const end = Math.max(buffers.current.get(BACKING)?.duration ?? 0, buffers.current.get(TARGET)?.duration ?? 0);
    return { key: "whole", t0: 0, t1: end || cat.lines[cat.lines.length - 1].t1 + 4 };
  }, [cat, loaded]); // `loaded` counts the decoded files: recompute as they arrive
  const name = (id: string) => (id === TARGET ? "Singer" : blind ? `Voice ${LETTERS[order.indexOf(id)]}` : (byId.get(id)?.label ?? id));
  const short = (id: string) => (id === TARGET ? "Singer" : blind ? LETTERS[order.indexOf(id)] : String(order.indexOf(id) + 1));
  const performer = shown && teles.has(shown) ? shown : (order.find((id) => teles.has(id)) ?? null);

  const stop = useCallback(() => {
    if (queue.current !== null) clearTimeout(queue.current);
    queue.current = null;
    if (current.current) {
      current.current.onended = null;
      current.current.stop();
      current.current = null;
    }
    musicSrc.current?.stop();
    musicSrc.current = null;
    clockRef.current = null;
    setPlaying(null);
  }, []);

  const play = useCallback(
    (id: string, span: Span, then?: () => void) => {
      const b = buffers.current.get(id), t = buffers.current.get(TARGET);
      if (!b || !t) return;
      stop();
      if (id !== TARGET) setShown(id);
      const ac = audio();
      void ac.resume();
      const src = ac.createBufferSource();
      src.buffer = b;
      const gain = ac.createGain();
      gain.gain.value = id === TARGET ? 1 : rms(t, span.t0, span.t1) / rms(b, span.t0, span.t1);
      src.connect(gain).connect(ac.destination);
      src.onended = () => {
        current.current = null;
        musicSrc.current?.stop();
        musicSrc.current = null;
        clockRef.current = null;
        setPlaying(null);
        if (then) queue.current = window.setTimeout(then, 350);
      };
      current.current = src;
      setPlaying({ id, span: span.key });
      // voice and band start on the same sample of the same timeline
      const at = ac.currentTime + 0.03;
      const band = buffers.current.get(BACKING);
      if (music && band) {
        const m = ac.createBufferSource();
        m.buffer = band;
        m.connect(ac.destination);
        m.start(at, span.t0, span.t1 - span.t0);
        musicSrc.current = m;
      }
      src.start(at, span.t0, span.t1 - span.t0);
      clockRef.current = { at, t0: span.t0, t1: span.t1 };
      epoch.current++;
    },
    [audio, stop, music],
  );

  const playLine = useCallback(
    (id: string, li: number, then?: () => void) => {
      if (!spans[li]) return;
      setLine(li);
      play(id, spans[li], then);
    },
    [spans, play],
  );

  const playAll = useCallback(
    (li: number) => {
      const ids = [TARGET, ...order];
      const step = (k: number) => k < ids.length && playLine(ids[k], li, () => step(k + 1));
      step(0);
    },
    [order, playLine],
  );

  /** the playhead on the excerpt's timeline, or null */
  const now = useCallback((): number | null => {
    const c = clockRef.current, ac = ctx.current;
    if (!c || !ac) return null;
    return Math.min(c.t1, c.t0 + Math.max(0, ac.currentTime - c.at));
  }, []);

  // the time readout while something plays
  useEffect(() => {
    const id = setInterval(() => clockRef.current && setTick((n) => n + 1), 250);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!cat) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        setLine((l) => Math.min(cat.lines.length - 1, Math.max(0, l + (e.key === "ArrowDown" ? 1 : -1))));
      } else if (e.key === "0" || e.key.toLowerCase() === "c") playLine(TARGET, line);
      else if (/^[1-9]$/.test(e.key) && order[Number(e.key) - 1]) playLine(order[Number(e.key) - 1], line);
      else if (e.key === " ") {
        e.preventDefault();
        playAll(line);
      } else if (e.key === "Escape") stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cat, line, order, playLine, playAll, stop]);

  useEffect(() => () => stop(), [stop]);

  // the words at the mouth, as the singer sang them (the aligner's word times)
  const words = useMemo<SungWords | null>(() => {
    const all = cat?.lines.flatMap((l) => l.words ?? []) ?? [];
    if (!all.length) return null;
    return { times: all.map(([, a, b]) => [a, b] as [number, number]), shown: all.map(([w]) => w), parent: all.map((_, i) => i) };
  }, [cat]);
  const overlay = useMemo(() => (words ? mouthWords(() => (clockRef.current ? wordAt(words, now() ?? -1) : null), V.green, 1) : undefined), [words, now]);

  // the performer: one feed per shown voice, reading the playhead each frame
  const feed = useMemo<HeadFeed | null>(() => {
    const c = performer ? byId.get(performer) : undefined;
    const track = performer ? teles.get(performer) : undefined;
    if (!c || !track) return null;
    const model = getModel(c.body ?? track.body, track.sampleRate);
    let last = VoiceModel.articFromArray(track.message(0).t.artic);
    const strums = cat?.strums ?? [];
    // the left hand climbs the neck with the sung pitch (eased, held through unvoiced sounds)
    let fret = 0.3;
    return {
      model,
      view: VIEW,
      stage: true,
      frame: () => {
        const t = now();
        if (t === null) return { tele: null, artic: last, epoch: epoch.current, pose: { strum: 1, fret } };
        const msg = track.message(track.indexAt(t));
        last = VoiceModel.articFromArray(msg.t.artic);
        if (msg.t.f0 > 40 && msg.t.voicing > 0.3) fret += (Math.min(1, Math.max(0, Math.log2(msg.t.f0 / 85) / 1.1)) - fret) * 0.08;
        return { tele: msg, artic: last, epoch: epoch.current, pose: { strum: strums.length ? strumAt(strums, t) : 0, fret } };
      },
    };
  }, [performer, byId, teles, now, cat]);

  if (error)
    return (
      <div className="vf">
        <Screen title="Voice fit" sub="not loaded">
          <p className="vf-msg">
            {error}. Build the catalog with <code>tools/voicefit/catalog.py &lt;fit dir&gt;</code> and point the dev server at the folder that holds it (<code>VOICEFIT_DIR</code>, or a link at <code>out/voicefit</code>).
          </p>
        </Screen>
      </div>
    );
  if (!cat) return <div className="vf" />;
  const total = cat.candidates.length + 1 + (cat.accompaniment ? 1 : 0);
  const t = now();

  const buttons = (span: Span, onPlay: (id: string) => void) =>
    [TARGET, ...order].map((id) => (
      <Chip
        key={id}
        className={id === TARGET ? "vf-singer" : undefined}
        on={playing?.id === id && playing.span === span.key}
        toggle={false}
        title={`Play ${name(id)}`}
        onClick={(e) => (e.stopPropagation(), onPlay(id))}
      >
        {short(id)}
      </Chip>
    ));

  return (
    <div className="vf">
      <Screen
        className="vf-stage"
        title="Stage"
        sub={`${playing ? `${name(playing.id)} · ${clock(t ?? 0)}` : "press play"}${performer ? ` · moves of ${name(performer)}` : ""}`}
        tools={
          sets.length > 1 ? (
            <>
              {sets.map((s) => (
                <Chip key={s} on={s === set} onClick={() => (window.location.search = `?voicefit=${s}`)}>
                  {s}
                </Chip>
              ))}
            </>
          ) : undefined
        }
      >
        {feed ? (
          <Suspense fallback={null}>
            <AnatomyView3D key={performer ?? ""} feed={feed} readouts={false} overlay={overlay} />
          </Suspense>
        ) : (
          <p className="vf-msg">{teles.size ? "" : "Loading the performer…"}</p>
        )}
      </Screen>

      <Screen
        title="Voice fit"
        sub={`${cat.title} · ${loaded < total ? `loading ${loaded}/${total}` : "↑↓ line · 0 singer · 1-9 voice · space all · esc stop"}`}
        tools={
          <>
            {cat.accompaniment && (
              <Chip on={music} onClick={() => setMusic((m) => !m)} title="Play the band under every voice, at the record's balance">
                {music ? "✓ Backing music" : "Backing music off"}
              </Chip>
            )}
            <Chip on={!blind} onClick={() => setBlind((b) => !b)}>
              {blind ? "Reveal" : "Revealed"}
            </Chip>
            <Chip toggle={false} onClick={() => playAll(line)}>
              Play line through all
            </Chip>
            <Chip toggle={false} onClick={stop}>
              Stop
            </Chip>
          </>
        }
      >
        {whole && (
          <section className="vf-verse">
            <span className="vf-text">{cat.lines.length > 10 ? "Whole song" : "Whole excerpt"}</span>
            <span className="vf-split">{clock(whole.t1 - whole.t0)}</span>
            <span className="vf-plays">{buttons(whole, (id) => play(id, whole))}</span>
          </section>
        )}
        <ol className="vf-lines">
          {cat.lines.map((l, li) => (
            <li key={l.i} className={li === line ? "vf-line on" : "vf-line"} onClick={() => setLine(li)}>
              <span className="vf-text">{l.text}</span>
              <span className="vf-split">{l.split === "train" ? "fitted on" : l.split === "heldout" ? "held out" : ""}</span>
              <span className="vf-plays">{buttons(spans[li], (id) => playLine(id, li))}</span>
            </li>
          ))}
        </ol>
      </Screen>

      {!blind && (
        <Screen title="Voices" sub="lower score is closer; similarity and word error never steered a fit, except on the fitted lines in rounds 4 and 5">
          <div className="vf-table-wrap">
            <table className="vf-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Voice</th>
                  <th>In the app</th>
                  <th>Score fitted</th>
                  <th>Score held out</th>
                  <th>Like him (held out)</th>
                  <th>Word error (held out)</th>
                </tr>
              </thead>
              <tbody>
                {order.map((id, i) => {
                  const c = byId.get(id)!;
                  return (
                    <tr key={id} title={c.about}>
                      <td>{i + 1}</td>
                      <td>{c.label}</td>
                      <td>{voiceById(c.voice ?? null)?.name ?? ""}</td>
                      <td>{c.scores.train?.score.toFixed(2)}</td>
                      <td>{c.scores.heldout?.score.toFixed(2)}</td>
                      <td>{c.judges.heldout?.similarity.toFixed(3)}</td>
                      <td>{c.judges.heldout ? pct(c.judges.heldout.wer) : ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Screen>
      )}
    </div>
  );
}
