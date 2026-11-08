/**
 * The voice-fit comparison page (`?voicefit`, dev server only): every candidate voice from
 * tools/voicefit against the recording it was fitted to, the whole verse or line by line.
 *
 * Blind by default: candidates get letters in an order shuffled once per browser, and their names
 * and measures stay hidden until Reveal. Every play is level-matched to the singer over the same
 * stretch, and by default the band (the separated instrumental) plays under it at the record's
 * own balance, to hear how each voice sits in the song.
 *
 * Keys: up/down pick a line, 0 or C plays the singer, 1-9 the candidates, space the line through
 * all of them, Esc stops.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chip, Screen } from "./Screen.tsx";
import { voiceById } from "../../voices/index.ts";

type Split = "train" | "heldout";

interface Line {
  i: number;
  text: string;
  t0: number;
  t1: number;
  split: Split | null;
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
  lines: Line[];
  candidates: Candidate[];
}

/** a stretch of the timeline: one line, or the whole verse */
interface Span {
  key: string;
  t0: number;
  t1: number;
}

const BASE = `${import.meta.env.BASE_URL}voicefit/`;
const TARGET = "target";
const BACKING = "backing";
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

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

export function VoicefitPage() {
  const [cat, setCat] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [blind, setBlind] = useState(true);
  // the band under every rendition, at the song's own balance: how a voice sits in the record
  const [music, setMusic] = useState(true);
  const [line, setLine] = useState(0);
  /** what is playing: voice id and span key */
  const [playing, setPlaying] = useState<{ id: string; span: string } | null>(null);
  const [loaded, setLoaded] = useState(0);
  const ctx = useRef<AudioContext | null>(null);
  const buffers = useRef(new Map<string, AudioBuffer>());
  const current = useRef<AudioBufferSourceNode | null>(null);
  const musicSrc = useRef<AudioBufferSourceNode | null>(null);
  const queue = useRef<number | null>(null);

  useEffect(() => {
    fetch(BASE + "candidates.json", { cache: "no-store" })
      .then(async (r) => {
        if (!r.ok) throw new Error(`candidates.json: ${r.status} ${await r.text()}`);
        setCat((await r.json()) as Catalog);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const audio = useCallback(() => (ctx.current ??= new AudioContext()), []);

  // decode the recording, the band and every render once
  useEffect(() => {
    if (!cat) return;
    const files: [string, string][] = [[TARGET, cat.target], ...cat.candidates.map((c) => [c.id, c.wav] as [string, string])];
    if (cat.accompaniment) files.push([BACKING, cat.accompaniment]);
    let alive = true;
    void Promise.all(
      files.map(async ([id, file]) => {
        const r = await fetch(BASE + file);
        if (!r.ok) throw new Error(`${file}: ${r.status}`);
        const b = await audio().decodeAudioData(await r.arrayBuffer());
        if (!alive) return;
        buffers.current.set(id, b);
        setLoaded((n) => n + 1);
      }),
    ).catch((e) => setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [cat, audio]);

  const order = useMemo(() => (cat ? (blind ? blindOrder(cat.candidates.map((c) => c.id)) : cat.candidates.map((c) => c.id)) : []), [cat, blind]);
  const byId = useMemo(() => new Map(cat?.candidates.map((c) => [c.id, c]) ?? []), [cat]);
  const spans = useMemo<Span[]>(() => cat?.lines.map((l) => ({ key: `line${l.i}`, t0: l.t0, t1: l.t1 })) ?? [], [cat]);
  const verse = useMemo<Span | null>(() => (cat && cat.lines.length ? { key: "verse", t0: cat.lines[0].t0, t1: cat.lines[cat.lines.length - 1].t1 } : null), [cat]);
  const name = (id: string) => (id === TARGET ? "Singer" : blind ? `Voice ${LETTERS[order.indexOf(id)]}` : (byId.get(id)?.label ?? id));
  const short = (id: string) => (id === TARGET ? "Singer" : blind ? LETTERS[order.indexOf(id)] : String(order.indexOf(id) + 1));

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
    setPlaying(null);
  }, []);

  const play = useCallback(
    (id: string, span: Span, then?: () => void) => {
      const b = buffers.current.get(id), t = buffers.current.get(TARGET);
      if (!b || !t) return;
      stop();
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

  if (error)
    return (
      <div className="vf">
        <Screen title="Voice fit" sub="not loaded">
          <p className="vf-msg">
            {error}. Build the catalog with <code>tools/voicefit/catalog.py &lt;fit dir&gt;</code> and point the dev server at that folder (<code>VOICEFIT_DIR</code>, or a link at <code>out/voicefit</code>).
          </p>
        </Screen>
      </div>
    );
  if (!cat) return <div className="vf" />;
  const total = cat.candidates.length + 1 + (cat.accompaniment ? 1 : 0);

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
        {verse && (
          <section className="vf-verse">
            <span className="vf-text">Whole verse</span>
            <span className="vf-split">{Math.round(verse.t1 - verse.t0)} s</span>
            <span className="vf-plays">{buttons(verse, (id) => play(id, verse))}</span>
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
