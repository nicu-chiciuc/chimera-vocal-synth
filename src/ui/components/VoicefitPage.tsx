/**
 * The voice-fit comparison page (`?voicefit`, dev server only): every candidate voice from
 * tools/voicefit, line by line against the recording it was fitted to, for a listener to judge.
 *
 * Blind by default: candidates get letters in an order shuffled once per browser, and their names,
 * settings and measures stay hidden until Reveal. Every play is level-matched to the singer's
 * line. Ratings (sounds like the singer, words clear, a note) are saved to the fit's folder
 * (`ratings.json`, through the dev server: see vite.config.ts), where the next fit can read them.
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
  lines: Line[];
  candidates: Candidate[];
}

interface Rating {
  like?: number;
  clear?: number;
  note?: string;
  at?: string;
}

type Ratings = Record<string, Rating>;

const BASE = `${import.meta.env.BASE_URL}voicefit/`;
const TARGET = "target";
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

async function getJson<T>(name: string, fallback?: T): Promise<T> {
  const r = await fetch(BASE + name, { cache: "no-store" });
  if (!r.ok) {
    if (fallback !== undefined && r.status === 404) return fallback;
    throw new Error(`${name}: ${r.status} ${await r.text()}`);
  }
  return r.json() as Promise<T>;
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

export function VoicefitPage() {
  const [cat, setCat] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ratings, setRatings] = useState<Ratings>({});
  const [blind, setBlind] = useState(true);
  const [line, setLine] = useState(0);
  const [playing, setPlaying] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(0);
  const ctx = useRef<AudioContext | null>(null);
  const buffers = useRef(new Map<string, AudioBuffer>());
  const current = useRef<AudioBufferSourceNode | null>(null);
  const queue = useRef<number | null>(null);
  const saveTimer = useRef<number | null>(null);

  useEffect(() => {
    (async () => {
      const c = await getJson<Catalog>("candidates.json");
      setCat(c);
      setRatings(await getJson<Ratings>("ratings.json", {}));
    })().catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const audio = useCallback(() => (ctx.current ??= new AudioContext()), []);

  // decode the recording and every render once
  useEffect(() => {
    if (!cat) return;
    const files: [string, string][] = [[TARGET, cat.target], ...cat.candidates.map((c) => [c.id, c.wav] as [string, string])];
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
    setPlaying(null);
  }, []);

  const play = useCallback(
    (id: string, li: number, then?: () => void) => {
      const b = buffers.current.get(id), t = buffers.current.get(TARGET), l = cat?.lines[li];
      if (!b || !t || !l) return;
      stop();
      const ac = audio();
      void ac.resume();
      const src = ac.createBufferSource();
      src.buffer = b;
      const gain = ac.createGain();
      gain.gain.value = id === TARGET ? 1 : rms(t, l.t0, l.t1) / rms(b, l.t0, l.t1);
      src.connect(gain).connect(ac.destination);
      src.onended = () => {
        current.current = null;
        setPlaying(null);
        if (then) queue.current = window.setTimeout(then, 350);
      };
      current.current = src;
      setPlaying(id);
      setLine(li);
      src.start(0, l.t0, l.t1 - l.t0);
    },
    [cat, audio, stop],
  );

  const playAll = useCallback(
    (li: number) => {
      const ids = [TARGET, ...order];
      const step = (k: number) => k < ids.length && play(ids[k], li, () => step(k + 1));
      step(0);
    },
    [order, play],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!cat || (e.target instanceof HTMLElement && e.target.closest("input, textarea"))) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        setLine((l) => Math.min(cat.lines.length - 1, Math.max(0, l + (e.key === "ArrowDown" ? 1 : -1))));
      } else if (e.key === "0" || e.key.toLowerCase() === "c") play(TARGET, line);
      else if (/^[1-9]$/.test(e.key) && order[Number(e.key) - 1]) play(order[Number(e.key) - 1], line);
      else if (e.key === " ") {
        e.preventDefault();
        playAll(line);
      } else if (e.key === "Escape") stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cat, line, order, play, playAll, stop]);

  useEffect(() => () => stop(), [stop]);

  const rate = (id: string, r: Rating) => {
    setRatings((prev) => {
      const next = { ...prev, [id]: { ...prev[id], ...r, at: new Date().toISOString() } };
      if (saveTimer.current !== null) clearTimeout(saveTimer.current);
      saveTimer.current = window.setTimeout(() => {
        void fetch(BASE + "ratings.json", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(next, null, 1) }).catch((e) => setError(String(e)));
      }, 400);
      return next;
    });
  };

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
  const total = cat.candidates.length + 1;

  return (
    <div className="vf">
      <Screen
        title="Voice fit"
        sub={`${cat.title} · ${loaded < total ? `loading ${loaded}/${total}` : "↑↓ line · 0 singer · 1-9 voice · space all · esc stop"}`}
        tools={
          <>
            <Chip on={!blind} onClick={() => setBlind((b) => !b)}>
              {blind ? "Reveal" : "Revealed"}
            </Chip>
            <Chip toggle={false} onClick={() => playAll(line)}>
              Play line through all
            </Chip>
          </>
        }
      >
        <ol className="vf-lines">
          {cat.lines.map((l, li) => (
            <li key={l.i} className={li === line ? "vf-line on" : "vf-line"} onClick={() => setLine(li)}>
              <span className="vf-text">{l.text}</span>
              <span className="vf-split">{l.split === "train" ? "fitted on" : l.split === "heldout" ? "held out" : ""}</span>
              <span className="vf-plays">
                {[TARGET, ...order].map((id) => (
                  <Chip key={id} on={playing === id && li === line} toggle={false} title={`Play ${name(id)}`} onClick={(e) => (e.stopPropagation(), play(id, li))}>
                    {id === TARGET ? "Singer" : short(id)}
                  </Chip>
                ))}
              </span>
            </li>
          ))}
        </ol>
      </Screen>

      <Screen title="Your ratings" sub="saved to the fit's folder as you go">
        <div className="vf-cands">
          {order.map((id) => {
            const c = byId.get(id)!;
            const r = ratings[id] ?? {};
            const v = voiceById(c.voice ?? null);
            return (
              <section key={id} className="vf-cand">
                <h3 className="vf-name">
                  {name(id)}
                  {!blind && v && <span className="vf-voice">in the app: Voice → {v.name}</span>}
                </h3>
                {!blind && c.about && <p className="vf-about">{c.about}</p>}
                <div className="vf-rate">
                  <span>Sounds like him</span>
                  {[1, 2, 3, 4, 5].map((n) => (
                    <Chip key={n} on={r.like === n} onClick={() => rate(id, { like: n })}>
                      {n}
                    </Chip>
                  ))}
                </div>
                <div className="vf-rate">
                  <span>Words clear</span>
                  {[1, 2, 3, 4, 5].map((n) => (
                    <Chip key={n} on={r.clear === n} onClick={() => rate(id, { clear: n })}>
                      {n}
                    </Chip>
                  ))}
                </div>
                <input className="vf-note" placeholder="Note (what's off?)" value={r.note ?? ""} onChange={(e) => rate(id, { note: e.target.value })} />
              </section>
            );
          })}
        </div>
      </Screen>

      {!blind && (
        <Screen title="Measures" sub="lower score is closer; similarity and word error never steered the fit (except round 4's training lines)">
          <div className="vf-table-wrap">
            <table className="vf-table">
              <thead>
                <tr>
                  <th>Voice</th>
                  <th>Score fitted</th>
                  <th>Score held out</th>
                  <th>Like him (held out)</th>
                  <th>Word error (held out)</th>
                  <th>Your like</th>
                  <th>Your clear</th>
                </tr>
              </thead>
              <tbody>
                {order.map((id) => {
                  const c = byId.get(id)!;
                  return (
                    <tr key={id}>
                      <td>{c.label}</td>
                      <td>{c.scores.train?.score.toFixed(2)}</td>
                      <td>{c.scores.heldout?.score.toFixed(2)}</td>
                      <td>{c.judges.heldout?.similarity.toFixed(3)}</td>
                      <td>{c.judges.heldout ? pct(c.judges.heldout.wer) : ""}</td>
                      <td>{ratings[id]?.like ?? ""}</td>
                      <td>{ratings[id]?.clear ?? ""}</td>
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
