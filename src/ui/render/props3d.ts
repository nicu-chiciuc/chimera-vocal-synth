/**
 * Stage props for the 3D head (`Anatomy3D.stage`): a wide-brimmed hat on the head, and under it a
 * whole performer standing on a stage. A torso and legs, arms with hands whose fingers fret and
 * strum an acoustic guitar, a microphone on a straight stand in front of the mouth, and the stage
 * itself: a platform, a curtain behind, a pool of light at the feet. Glowing lines like the head.
 *
 * Body cm, the head's frame: x forward, y up, z out of the right side of the face, glottis at the
 * origin. The hat is placed from the head's own cross-sections, so it fits any body; the torso
 * hangs from the bottom of the neck. The floor, the stage and the microphone stand a fixed
 * distance below and in front of the face, so on a stage (which puts every face on one point) they
 * stay put when the voice changes; the legs take up the difference. The hat nods with the head;
 * nothing else does (the renderer undoes the head's turn for the rest).
 *
 * The hands are the moving part. The left hand holds a chord shape, picked by the pitch, its
 * fingers placed on the strings by two-bone inverse kinematics from the knuckles; the right hand
 * holds a pick and swings from the wrist across the strings. The body sways from the hips, the
 * right toe taps, and the strings shiver after a strum.
 */
import { V } from "../colors.ts";

type V3 = [number, number, number];

/** what the props need of the head: its top and bottom, and a cross-section at a height */
export interface HeadSections {
  y0: number;
  y1: number;
  at(y: number): { F: number; Bk: number; W: number } | null;
}

export interface PropGroup {
  lines: Float32Array[];
  color: string;
  width: number;
  alpha: number;
}

/** the player this frame */
export interface Pose {
  /** the right hand across the strings: 1 above the low string, −1 below the high one */
  strum: number;
  /** the left hand's place on the neck: 0 at the nut, 1 high up towards the body */
  fret: number;
  /** the right toe off the floor, 0..1 */
  tap: number;
  /** seconds since the strings were last struck (Infinity: not yet) */
  pluck: number;
}

export interface Props {
  /** carried by the head (it nods with it) */
  hat: PropGroup[];
  /** the stage and the microphone: still */
  scene: PropGroup[];
  /** the performer below the head this frame; `dt` is the time since the last frame (s) */
  frame(pose: Pose, dt: number): PropGroup[];
}

class Lines {
  readonly lines: Float32Array[] = [];
  private cur: number[] = [];
  p(q: V3): this {
    this.cur.push(q[0], q[1], q[2]);
    return this;
  }
  end(close = false): void {
    if (close && this.cur.length >= 3) this.cur.push(this.cur[0], this.cur[1], this.cur[2]);
    if (this.cur.length >= 6) this.lines.push(new Float32Array(this.cur));
    this.cur = [];
  }
  path(pts: V3[], close = false): void {
    for (const q of pts) this.p(q);
    this.end(close);
  }
}

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => mul(a, 1 / (len(a) || 1));
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
/** `a` with its part along unit `d` removed, as a unit vector */
const perp = (a: V3, d: V3): V3 => norm(sub(a, mul(d, dot(a, d))));
/** a sum of scaled vectors */
const lin = (...terms: [V3, number][]): V3 => terms.reduce<V3>((acc, [v, s]) => add(acc, mul(v, s)), [0, 0, 0]);

function interp(tab: readonly (readonly [number, number])[], x: number): number {
  if (x <= tab[0][0]) return tab[0][1];
  for (let i = 1; i < tab.length; i++) {
    if (x <= tab[i][0]) {
      const [x0, y0] = tab[i - 1], [x1, y1] = tab[i];
      const t = (x - x0) / (x1 - x0);
      // smoothstep between the table's points: round bouts, no corners
      return y0 + (y1 - y0) * t * t * (3 - 2 * t);
    }
  }
  return tab[tab.length - 1][1];
}

/** a tube along a path: rings at every `ringEvery`th point, and `sides` lines along it. The
 *  rings' frames are carried along the path (parallel transport), so the lines never twist. */
function tube(L: Lines, path: V3[], radius: (i: number) => number, sides = 6, ringEvery = 1, ringN = 16): void {
  const n = path.length;
  if (n < 2) return;
  const frames: { p: V3; u: V3; v: V3 }[] = [];
  let u: V3 | null = null;
  for (let i = 0; i < n; i++) {
    const t = norm(sub(path[Math.min(n - 1, i + 1)], path[Math.max(0, i - 1)]));
    if (!u) {
      const up: V3 = Math.abs(t[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
      u = norm(cross(up, t));
    } else u = perp(u, t);
    frames.push({ p: path[i], u, v: cross(t, u) });
  }
  const at = (i: number, a: number): V3 => {
    const f = frames[i], r = radius(i);
    return add(f.p, add(mul(f.u, r * Math.cos(a)), mul(f.v, r * Math.sin(a))));
  };
  for (let i = 0; i < n; i += ringEvery) {
    for (let s = 0; s <= ringN; s++) L.p(at(i, (s / ringN) * Math.PI * 2));
    L.end();
  }
  for (let s = 0; s < sides; s++) {
    const a = (s / sides) * Math.PI * 2;
    for (let i = 0; i < n; i++) L.p(at(i, a));
    L.end();
  }
}

/** a smooth path through control points (Catmull-Rom, `n` steps per span) */
function spline(pts: V3[], n = 6): V3[] {
  const out: V3[] = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    for (let j = 0; j < n; j++) {
      const t = j / n, t2 = t * t, t3 = t2 * t;
      out.push([0, 1, 2].map((c) => 0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2 + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3)) as V3);
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}

/** the middle joint of two bones from `a` to `b` (an elbow, a knee, a knuckle): bent towards
 *  `bend`, and straightened when `b` is out of reach */
function ik2(a: V3, b: V3, l1: number, l2: number, bend: V3): V3 {
  const d0 = sub(b, a);
  const d = Math.max(Math.abs(l1 - l2) + 0.01, Math.min(len(d0), (l1 + l2) * 0.995));
  const dir = norm(d0);
  const x = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, l1 * l1 - x * x));
  return add(a, add(mul(dir, x), mul(perp(bend, dir), h)));
}

/** a circle of `n` points about `c` in the plane of unit `u`, `v` */
function circle(c: V3, u: V3, v: V3, r: number, n = 32): V3[] {
  const pts: V3[] = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push(add(c, add(mul(u, r * Math.cos(a)), mul(v, r * Math.sin(a)))));
  }
  return pts;
}

// ------------------------------------------------------------------ the hat

function hat(head: HeadSections, k: number): PropGroup[] {
  // the hat sits on the head where it is widest above the ears, and covers the crown
  const yb = head.y1 - 5.2 * k;
  const s = head.at(yb) ?? { F: 8 * k, Bk: -10 * k, W: 7.4 * k };
  const cx = (s.F + s.Bk) / 2 + 0.2 * k;
  const rx = (s.F - s.Bk) / 2 + 0.7 * k, rz = s.W + 0.7 * k;
  const crownH = 8.2 * k, brim = 7.0 * k;
  const ring = (y: (c: number, sn: number) => number, ax: number, az: number, n = 48): V3[] => {
    const pts: V3[] = [];
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * Math.PI * 2, c = Math.cos(a), sn = Math.sin(a);
      pts.push([cx + ax * c, y(c, sn), az * sn]);
    }
    return pts;
  };
  // the brim rolls up at the sides and dips a little at the front, as a cattleman's does
  const brimY = (c: number, sn: number, f: number) => yb + f * (2.2 * k * sn * sn - 0.6 * k * Math.max(0, c));
  // the crown narrows slightly to a top creased front to back, with a pinch at the front
  const crownTop = (c: number, sn: number) => yb + crownH - 1.5 * k * (1 - sn * sn) * 0.6 - 1.2 * k * Math.max(0, c) ** 3;
  const crown = new Lines(), band = new Lines(), brims = new Lines();
  for (const f of [0.25, 0.5, 0.75, 1]) {
    const taper = 1 - 0.07 * f;
    crown.path(ring((c, sn) => yb + 0.35 * k + f * (crownTop(c, sn) - yb - 0.35 * k), rx * taper, rz * taper));
  }
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2, c = Math.cos(a), sn = Math.sin(a);
    const pts: V3[] = [];
    for (let j = 0; j <= 8; j++) {
      const f = j / 8, taper = 1 - 0.07 * f;
      pts.push([cx + rx * taper * c, yb + 0.35 * k + f * (crownTop(c, sn) - yb - 0.35 * k), rz * taper * sn]);
    }
    crown.path(pts);
  }
  // the crease across the top
  const crease: V3[] = [];
  for (let j = -8; j <= 8; j++) {
    const c = j / 8;
    crease.push([cx + rx * 0.9 * c, crownTop(c, 0) - 0.5 * k, 0]);
  }
  crown.path(crease);
  // the band, just above the brim
  for (const h of [0.35, 1.5]) band.path(ring(() => yb + h * k, rx * 1.005, rz * 1.005));
  // the brim: its inner edge, middle and outer edge, and spokes between
  for (const f of [0, 0.5, 1]) brims.path(ring((c, sn) => brimY(c, sn, f), rx + f * brim, rz + f * brim * 0.95));
  for (let i = 0; i < 24; i++) {
    const a = (i / 24) * Math.PI * 2, c = Math.cos(a), sn = Math.sin(a);
    brims.path([0, 0.5, 1].map((f) => [cx + (rx + f * brim) * c, brimY(c, sn, f), (rz + f * brim * 0.95) * sn] as V3));
  }
  return [
    { lines: crown.lines, color: V.white, width: 1.05, alpha: 0.75 },
    { lines: band.lines, color: V.amber, width: 1.3, alpha: 0.9 },
    { lines: brims.lines, color: V.white, width: 1.05, alpha: 0.8 },
  ];
}

// ------------------------------------------------------------------ the guitar's measures

/** half-width of the guitar's body along its length (cm from the waist, + towards the neck) */
const GUITAR_W = [[-31, 0], [-30.2, 7], [-27.5, 14], [-22, 18.8], [-15, 19.6], [-8, 17], [-2.5, 12.6], [1.5, 12.2], [7, 13.8], [13, 14.6], [18, 12.4], [21.5, 7.5], [22.6, 0]] as const;
/** the nut and the bridge along the neck (cm from the waist), and the scale between them */
const NUT = 66, BRIDGE = -16, SCALE = NUT - BRIDGE;
/** fret f's wire along the neck */
const fretAt = (f: number) => NUT - SCALE * (1 - 2 ** (-f / 12));
/** where a finger presses for fret f: just behind the wire */
const pressAt = (f: number) => fretAt(f) + 0.3 * (fretAt(f - 1) - fretAt(f));
/** string s across the neck at `a` (0 the high e at the bottom, 5 the low E on top) */
const stringB = (s: number, a: number) => lerp(-3.2 + (6.4 * s) / 5, -2.0 + (4.0 * s) / 5, (a - BRIDGE) / SCALE);
/** the strings' height over the soundboard */
const STRING_C = 5.75;

// ------------------------------------------------------------------ the hands' measures

/** finger bones, index to little: proximal, middle, distal (cm) */
const FINGER = [[4.5, 2.6, 2.2], [4.9, 3.0, 2.4], [4.6, 2.8, 2.3], [3.6, 2.1, 1.9]] as const;
/** the knuckles across the palm, thumb side first (cm) */
const KNUCKLE = [2.8, 0.95, -0.95, -2.8] as const;
const THUMB = [4.5, 3.4] as const;

interface Press {
  s: number;
  f: number;
}
/** chord shapes the left hand moves between: whether the index bars the fret, and where the other
 *  fingers press (string, frets above the base fret); null hovers */
const SHAPES: { barre: boolean; fingers: (Press | null)[] }[] = [
  { barre: true, fingers: [null, { s: 2, f: 1 }, { s: 4, f: 2 }, { s: 3, f: 2 }] },
  { barre: false, fingers: [{ s: 1, f: 0 }, { s: 3, f: 1 }, { s: 4, f: 2 }, null] },
  { barre: true, fingers: [null, { s: 3, f: 2 }, { s: 2, f: 2 }, { s: 1, f: 2 }] },
  { barre: true, fingers: [null, null, { s: 4, f: 2 }, { s: 3, f: 2 }] },
];

/** a hand's frame at the wrist: `h` towards the knuckles, `N` out of the back of the hand, `S`
 *  towards the thumb */
interface HandFrame {
  W: V3;
  h: V3;
  N: V3;
  S: V3;
}

/** the palm: two outlines (back and palm side) and the edges between */
function palm(L: Lines, f: HandFrame, Kc: V3, k: number): void {
  const c = [lin([f.W, 1], [f.S, 3.3 * k]), lin([Kc, 1], [f.S, 3.7 * k]), lin([Kc, 1], [f.S, -3.7 * k]), lin([f.W, 1], [f.S, -3.3 * k])];
  for (const sd of [1, -1]) L.path(c.map((q) => add(q, mul(f.N, sd * 1.1 * k))), true);
  for (const q of c) L.path([add(q, mul(f.N, 1.1 * k)), add(q, mul(f.N, -1.1 * k))]);
}

/** a finger as a thin tube through its joints */
function finger(L: Lines, pts: V3[], k: number, steps: number): void {
  const path = spline(pts, steps);
  const n = path.length - 1;
  tube(L, path, (i) => k * lerp(0.9, 0.62, i / n), 5, steps, 10);
}

/** a finger curled by its joint angles (towards the palm), from knuckle `K` */
function curled(f: HandFrame, K: V3, bones: readonly [number, number, number], angles: [number, number, number], k: number): V3[] {
  const pts: V3[] = [K];
  let ang = 0, p = K;
  for (let i = 0; i < 3; i++) {
    ang += angles[i];
    p = lin([p, 1], [f.h, bones[i] * k * Math.cos(ang)], [f.N, -bones[i] * k * Math.sin(ang)]);
    pts.push(p);
  }
  return pts;
}

/** a finger reaching from knuckle `K` to tip `T`, its middle knuckle bent towards `bend` */
function reaching(K: V3, T: V3, bones: readonly [number, number, number], bend: V3, k: number): V3[] {
  return [K, ik2(K, T, bones[0] * k, (bones[1] + bones[2]) * 0.93 * k, bend), T];
}

// ------------------------------------------------------------------ the performer and the stage

/** the stage's measures below the face (cm, before the body's scale) */
const FLOOR = 160;

function performer(head: HeadSections, k: number, face: V3): { scene: PropGroup[]; frame: Props["frame"] } {
  const yn = head.y0 + 1.0 * k;
  const ns = head.at(yn) ?? { F: 3 * k, Bk: -4 * k, W: 5.3 * k };
  const cx = (ns.F + ns.Bk) / 2 - 0.5 * k;
  /** the floor */
  const yF = face[1] - FLOOR * k;
  /** a point from the base of the neck (cm) */
  const B = (x: number, y: number, z: number): V3 => [cx + x * k, yn + y * k, z * k];
  /** a point from the face (cm) */
  const Fp = (x: number, y: number, z: number): V3 => [face[0] + x * k, face[1] + y * k, face[2] + z * k];
  /** a point on the floor (cm from under the neck) */
  const G = (x: number, z: number, y = 0): V3 => [cx + x * k, yF + y * k, z * k];

  // ---- the torso: cross-sections down from the base of the neck to the hips (y offset,
  // half-width, half-depth)
  const T: [number, number, number][] = [[0, 5.6, 5.0], [-3, 13, 8.5], [-6.5, 19, 10.5], [-12, 18.5, 11.5], [-22, 16.5, 12], [-34, 15.5, 11.5], [-46, 15.8, 11], [-52, 16.2, 11.4], [-58, 13.8, 9.8]];
  const at = (i: number, a: number): V3 => {
    const [dy, w, d] = T[i];
    // the shoulders fall away from the neck
    const drop = i <= 2 ? -2.6 * (Math.abs(Math.sin(a)) ** 2) * (i / 2) : 0;
    return B(d * Math.cos(a), dy + drop, w * Math.sin(a));
  };
  const torso = new Lines();
  for (let i = 0; i < T.length; i++) {
    for (let s = 0; s <= 32; s++) torso.p(at(i, (s / 32) * Math.PI * 2));
    torso.end();
  }
  for (let s = 0; s < 16; s++) {
    const a = (s / 16) * Math.PI * 2;
    torso.path(spline(T.map((_, i) => at(i, a)), 4));
  }

  // ---- the legs: standing, the left foot a little forward, the right one back and turned out
  const hip = (side: number) => B(0.5, -57, side * 9);
  const knee = (side: number) => B(side < 0 ? 5 : 2.5, -100, side * 11.5);
  const ankle = (side: number): V3 => [cx + (side < 0 ? 6 : -1) * k, yF + 7.5 * k, side * 13.2 * k];
  const legs = new Lines();
  for (const side of [-1, 1]) {
    const path = spline([hip(side), knee(side), ankle(side)], 6);
    tube(legs, path, (i) => k * (i <= 6 ? lerp(8, 6.2, i / 6) : lerp(6.2, 4.3, (i - 6) / 6)), 8, 3, 16);
  }
  /** the shoe: sole, upper, the ankle's ring and the instep; `alpha` lifts the toe about the heel */
  const SHOE_W = [[0, 0], [1, 2.6], [3.5, 3.8], [8.5, 4.3], [14.5, 4.5], [19.5, 4.3], [23, 3.2], [25, 1.2], [25.5, 0]] as const;
  const shoe = (L: Lines, side: number, alpha: number) => {
    const A = ankle(side), phi = side * 0.16;
    const fd: V3 = [Math.cos(phi), 0, Math.sin(phi)], sd: V3 = [-Math.sin(phi), 0, Math.cos(phi)];
    const heel: V3 = [A[0] - 4.5 * k * fd[0], yF, A[2] - 4.5 * k * fd[2]];
    const at = (s: number, w: number, y: number): V3 => {
      const ss = s * Math.cos(alpha) - y * Math.sin(alpha), yy = y * Math.cos(alpha) + s * Math.sin(alpha);
      return [heel[0] + (ss * fd[0] + w * sd[0]) * k, yF + yy * k, heel[2] + (ss * fd[2] + w * sd[2]) * k];
    };
    const loop = (y: number, shrink: number, s0: number, s1: number) => {
      const pts: V3[] = [];
      for (let s = s0; s <= s1; s += 1) pts.push(at(s, interp(SHOE_W, s) * shrink, y));
      for (let s = s1; s >= s0; s -= 1) pts.push(at(s, -interp(SHOE_W, s) * shrink, y));
      L.path(pts, true);
    };
    loop(0, 1, 0, 25.5);
    loop(2.8, 0.88, 0.6, 24);
    for (const s of [0.6, 8, 16, 23]) for (const sd2 of [1, -1]) L.path([at(s, sd2 * interp(SHOE_W, s), 0), at(s, sd2 * interp(SHOE_W, s) * 0.88, 2.8)]);
    // the ankle, and the instep down to the toe
    const ring: V3[] = [];
    for (let i = 0; i <= 20; i++) {
      const a = (i / 20) * Math.PI * 2;
      ring.push(at(4.5 + 4.0 * Math.cos(a), 4.0 * Math.sin(a), 7.5));
    }
    L.path(ring);
    for (const w of [0, 2.4, -2.4]) L.path(spline([at(8.4, w * 0.6, 7.3), at(14, w, 5.2), at(20.5, w * 0.9, 2.8)], 3));
    L.path([at(0.6, 0, 7.4), at(0.4, 0, 2.8)]);
  };
  shoe(legs, -1, 0);

  // ---- the guitar's frame: its waist in front of the belly, the neck rising across to the
  // player's left (−z), the strings facing the audience (+x)
  const O: V3 = B(17.5, -30, 3);
  const u = norm([0.05, 0.42, -1]);
  const n = perp([1, 0, 0.12], u);
  const v = cross(n, u);
  /** a point in the guitar's frame: along the neck, across it, out of the soundboard (cm) */
  const P = (q: V3): V3 => add(O, add(mul(u, q[0] * k), add(mul(v, q[1] * k), mul(n, q[2] * k))));
  const g = new Lines(), frets = new Lines(), tails = new Lines();
  const outline = (c: number): V3[] => {
    const pts: V3[] = [];
    for (let a = -31; a <= 22.6; a += 1.1) pts.push(P([a, interp(GUITAR_W, a), c]));
    const back: V3[] = [];
    for (let a = 22.6; a >= -31; a -= 1.1) back.push(P([a, -interp(GUITAR_W, a), c]));
    return [...pts, ...back];
  };
  g.path(outline(5), true);
  g.path(outline(-5), true);
  for (let a = -30; a <= 22; a += 6.5) for (const sd of [1, -1]) g.path([P([a, sd * interp(GUITAR_W, a), 5]), P([a, sd * interp(GUITAR_W, a), -5])]);
  // the sound hole, its rosette, the bridge
  for (const r of [4.4, 5.1]) {
    const hole: V3[] = [];
    for (let i = 0; i <= 32; i++) hole.push(P([7 + r * Math.cos((i / 32) * Math.PI * 2), r * Math.sin((i / 32) * Math.PI * 2), 5.05]));
    g.path(hole);
  }
  g.path([P([-17, -7.5, 5.1]), P([-17, 7.5, 5.1]), P([-15, 7.5, 5.1]), P([-15, -7.5, 5.1])], true);
  // the neck and fretboard to the nut, then the headstock
  const fb = (a: number) => lerp(2.8, 2.3, (a - 21) / (NUT - 21));
  g.path([P([21, -fb(21), 5.4]), P([NUT, -fb(NUT), 5.4]), P([NUT, fb(NUT), 5.4]), P([21, fb(21), 5.4])]);
  g.path([P([22.6, -fb(22.6), 1.5]), P([NUT, -fb(NUT), 3.2]), P([NUT, fb(NUT), 3.2]), P([22.6, fb(22.6), 1.5])]);
  for (let f = 1; f <= 19; f++) {
    const a = fretAt(f);
    if (a < 21) break;
    frets.path([P([a, -fb(a), 5.45]), P([a, fb(a), 5.45])]);
  }
  const head3 = [[NUT, 2.6], [NUT + 4, 4.4], [NUT + 17, 4.0], [NUT + 18.5, 3.2]] as const;
  g.path([...head3.map(([a, b]) => P([a, b, 4])), ...[...head3].reverse().map(([a, b]) => P([a, -b, 4]))], true);
  for (let i = 0; i < 3; i++) for (const sd of [1, -1]) g.path([P([NUT + 6 + 4.6 * i, sd * 4.2, 4]), P([NUT + 6 + 4.6 * i, sd * 6.4, 4])]);
  // the strings past the nut to the tuners (the sounding part is drawn each frame)
  for (let s = 0; s < 6; s++) tails.path([P([NUT, stringB(s, NUT), STRING_C]), P([NUT + 14, stringB(s, NUT) * 1.2, 4.2])]);

  // ---- the stage: a platform with boards, a pool of light, a curtain behind
  const stage = new Lines(), boards = new Lines(), pool = new Lines(), rays = new Lines(), curtain = new Lines();
  const SX = 100, SZ = 135, SE = 3.5, SCX = 12;
  const edge = (y: number): V3[] => {
    const pts: V3[] = [];
    for (let i = 0; i <= 72; i++) {
      const a = (i / 72) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
      pts.push(G(SCX + SX * Math.sign(c) * Math.abs(c) ** (2 / SE), SZ * Math.sign(s) * Math.abs(s) ** (2 / SE), y));
    }
    return pts;
  };
  stage.path(edge(0));
  stage.path(edge(-20));
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
    const x = SCX + SX * Math.sign(c) * Math.abs(c) ** (2 / SE), z = SZ * Math.sign(s) * Math.abs(s) ** (2 / SE);
    stage.path([G(x, z, 0), G(x, z, -20)]);
  }
  for (let z = -126; z <= 126; z += 18) {
    const hx = SX * (1 - Math.abs(z / SZ) ** SE) ** (1 / SE);
    boards.path([G(SCX - hx, z, 0.05), G(SCX + hx, z, 0.05)]);
  }
  const up: V3 = [0, 1, 0], ex: V3 = [1, 0, 0], ez: V3 = [0, 0, 1];
  const poolC = G(10, 2, 0.1);
  for (const [rx, rz] of [[60, 68], [46, 52], [30, 34]]) {
    const pts: V3[] = [];
    for (let i = 0; i <= 48; i++) {
      const a = (i / 48) * Math.PI * 2;
      pts.push(add(poolC, [rx * k * Math.cos(a), 0, rz * k * Math.sin(a)]));
    }
    pool.path(pts);
  }
  // the light's cone, from a lamp high in front and to the right
  const lamp = G(70, 60, 330);
  for (let i = 0; i < 10; i++) {
    const a = (i / 10) * Math.PI * 2;
    rays.path([lamp, add(poolC, [60 * k * Math.cos(a), 0, 68 * k * Math.sin(a)])]);
  }
  // the curtain: folds hanging from a rail to the floor, curved round the back of the platform
  // so it stands behind the performer from the front and from either side
  const rail: V3[] = [];
  for (let i = 0; i <= 38; i++) {
    const a = Math.PI * (0.55 + 0.92 * (i / 38)), fold = 5 * Math.sin(i * 1.9);
    const x = SCX + (178 + fold) * Math.cos(a), z = (178 + fold) * Math.sin(a);
    curtain.path([G(x, z, 240), G(x, z, -20)]);
    rail.push(G(x, z, 240));
  }
  curtain.path(rail);

  // ---- the microphone on a straight stand, as on a club stage: a heavy round base, one pole
  // with its clutch at mid height, the mic in a clip at the top tilted up toward the mouth, and
  // its cable hanging in a loop down to the floor. The pole stands in front of the guitar and a
  // little to the performer's right, so from the camera's three-quarter view it passes beside
  // the mouth, not over it
  const stand = new Lines(), mic = new Lines(), grille = new Lines(), cable = new Lines();
  const mouth = Fp(4.6, -3.7, 0);
  const base = Fp(27, -FLOOR, 12);
  const top = Fp(27, -20, 12);
  // the mic points between the mouth and straight up: steeper than the line to the mouth, as a
  // singer who stands back with a guitar sets it
  const d = norm(add(mul(norm(sub(mouth, top)), 0.6), [0, 0.5, 0]));
  const du = perp([0, 1, 0], d), dv = cross(d, du);
  const handleEnd = add(top, mul(d, 1.5 * k));
  const M = add(top, mul(d, 17.5 * k));
  for (const [r, h] of [[14, 0.3], [13.2, 1.6], [9, 2.8], [4.6, 3.6]]) stand.path(circle(add(base, [0, h * k, 0]), ex, ez, r * k, 40));
  const CLUTCH = 86;
  tube(stand, [add(base, [0, 3.6 * k, 0]), add(base, [0, CLUTCH * k, 0])], () => 1.45 * k, 4, 1, 12);
  tube(stand, [add(base, [0, (CLUTCH - 1) * k, 0]), add(base, [0, (CLUTCH + 5.5) * k, 0])], () => 2.3 * k, 6, 1, 14);
  tube(stand, [add(base, [0, (CLUTCH + 5.5) * k, 0]), top], () => 1.05 * k, 4, 1, 12);
  // the clip: a short sleeve round the handle's end, on a knuckle at the top of the pole
  stand.path(circle(top, ex, ez, 1.6 * k, 12));
  tube(stand, [sub(handleEnd, mul(d, 1 * k)), add(handleEnd, mul(d, 4 * k))], () => 2.3 * k, 6, 1, 12);
  // the mic: a slim handle widening to the head
  tube(mic, [handleEnd, add(handleEnd, mul(d, 6 * k)), sub(M, mul(d, 3 * k))], (i) => k * lerp(1.3, 1.75, i / 2), 6, 1, 14);
  for (const [off, r] of [[0, 2.9], [1.7, 2.3], [-1.7, 2.4], [2.6, 1.2]]) grille.path(circle(add(M, mul(d, off * k)), du, dv, r * k, 24));
  // the cable: out of the handle, a loose loop down past the pole, across the floor
  const plug = sub(handleEnd, mul(d, 2.5 * k));
  cable.path(spline([
    plug, add(plug, [4 * k, -12 * k, 7 * k]), add(base, [7 * k, 70 * k, 10 * k]), add(base, [9 * k, 30 * k, 8 * k]),
    add(base, [13 * k, 0.4 * k, 11 * k]), add(base, [30 * k, 0.4 * k, 24 * k]), add(base, [55 * k, 0.4 * k, 28 * k]),
  ], 8));
  for (const a of [0, Math.PI / 2]) {
    const w = add(mul(du, Math.cos(a)), mul(dv, Math.sin(a)));
    const pts: V3[] = [];
    for (let i = 0; i <= 16; i++) {
      const t = -Math.PI / 2 + (i / 16) * Math.PI;
      pts.push(add(M, add(mul(d, 2.9 * k * Math.sin(t)), mul(w, 2.9 * k * Math.cos(t)))));
    }
    grille.path(pts);
  }

  const scene: PropGroup[] = [
    { lines: stage.lines, color: V.blueDim, width: 1.05, alpha: 0.9 },
    { lines: boards.lines, color: V.frame, width: 0.85, alpha: 0.9 },
    { lines: pool.lines, color: V.amber, width: 1.3, alpha: 0.26 },
    { lines: rays.lines, color: V.amber, width: 0.6, alpha: 0.08 },
    { lines: curtain.lines, color: V.frame, width: 0.85, alpha: 0.85 },
    { lines: stand.lines, color: V.blueMid, width: 0.85, alpha: 0.7 },
    { lines: mic.lines, color: V.blueMid, width: 0.85, alpha: 0.75 },
    { lines: grille.lines, color: V.amber, width: 1.05, alpha: 0.9 },
    { lines: cable.lines, color: V.blueDim, width: 0.9, alpha: 0.7 },
  ];

  // ---- the moving parts
  const shoulder = (side: number): V3 => B(-0.5, -7.5, side * 18.5);
  const armR = (nPts: number) => (i: number) => k * lerp(4.8, 3.0, i / (nPts - 1));
  /** the right forearm rests on the lower bout's top edge, over the tail */
  const REST: V3 = [-20, 20.5, 8.5];
  const fingerBend = norm([0.15, 0.35, 1]);
  /** the left hand's state: the chord shown, where the hand is along the neck, where each
   *  fingertip and the thumb are (guitar frame), all eased towards their targets */
  let F = 3, aHand = pressAt(3);
  const tips: V3[] = [0, 1, 2, 3].map((i) => [aHand - 1.9 * i, -0.5, 7.6]);
  let thumbTip: V3 = [aHand + 1.5, 1.2, 1.5];
  let clock = 0;
  const ease = (from: V3, to: V3, r: number): V3 => [lerp(from[0], to[0], r), lerp(from[1], to[1], r), lerp(from[2], to[2], r)];

  const frame = (pose: Pose, dt: number): PropGroup[] => {
    clock += dt;
    // the sway: the hips drift from side to side, the feet and the neck staying put
    const swing = 2.2 * k * Math.sin(clock * 2 * Math.PI * 0.17);
    const sway = (y: number) => swing * Math.sin(Math.PI * clamp01((yn - y) / (yn - yF)));
    const shear = (lines: Float32Array[]): Float32Array[] =>
      lines.map((l) => {
        const o = new Float32Array(l.length);
        for (let i = 0; i < l.length; i += 3) {
          o[i] = l[i];
          o[i + 1] = l[i + 1];
          o[i + 2] = l[i + 2] + sway(l[i + 1]);
        }
        return o;
      });
    // the guitar and the hands move whole with the belly
    const dz: V3 = [0, 0, sway(yn - 40 * k)];
    const shift = (lines: Float32Array[]): Float32Array[] =>
      lines.map((l) => {
        const o = new Float32Array(l);
        for (let i = 2; i < o.length; i += 3) o[i] += dz[2];
        return o;
      });
    const PG = (q: V3): V3 => add(P(q), dz);
    /** lines drawn in the guitar's frame (cm, unscaled: P applies the body's scale) → body cm */
    const toBody = (lines: Float32Array[]): Float32Array[] =>
      lines.map((l) => {
        const o = new Float32Array(l.length);
        for (let i = 0; i < l.length; i += 3) {
          const q = PG([l[i], l[i + 1], l[i + 2]]);
          o[i] = q[0];
          o[i + 1] = q[1];
          o[i + 2] = q[2];
        }
        return o;
      });

    // ---- the right hand (guitar frame): hanging from the wrist over the strings, the pick
    // between thumb and index; the strum swings it from the wrist and carries the wrist with it
    const strum = Math.max(-1, Math.min(1, pose.strum));
    const theta = 0.3 * strum;
    const hR: V3 = [Math.sin(theta), -Math.cos(theta), 0];
    const NR = perp([0, 0.25, 1], hR);
    const SR = norm(cross(NR, hR));
    const right: HandFrame = { W: [-7.5 + 1.0 * strum, 9.0 + 4 * strum, 0], h: hR, N: NR, S: SR };
    const KcR = lin([right.W, 1], [hR, 8.5]);
    const ANGLES: [number, number, number][] = [[0.72, 1.25, 0.5], [0.9, 1.15, 0.6], [1.0, 1.1, 0.65], [1.05, 1.0, 0.7]];
    const fingersR = FINGER.map((bones, i) => curled(right, lin([KcR, 1], [SR, KNUCKLE[i]], [hR, i === 3 ? -0.5 : 0]), bones, ANGLES[i], 1));
    const grip = lin([fingersR[0][2], 1], [SR, 0.45], [NR, -0.4]);
    const dp = norm(lin([hR, 0.7], [NR, -0.75]));
    const pickTip = add(grip, mul(dp, 2.6));
    // the pick's tip rides on the strings: lift the whole hand to put it there
    const rise: V3 = [0, 0, STRING_C - pickTip[2]];
    const liftR = (q: V3): V3 => add(q, rise);
    const thumbBase = lin([right.W, 1], [hR, 4], [SR, 4.6], [NR, -1.4]);
    const thumbEnd = lin([grip, 1], [NR, 0.6], [hR, 0.3]);
    const thumbR = [thumbBase, ik2(thumbBase, thumbEnd, THUMB[0], THUMB[1], lin([SR, 1], [NR, 0.4])), thumbEnd];
    const handR = new Lines(), pick = new Lines();
    palm(handR, { ...right, W: liftR(right.W) }, liftR(KcR), 1);
    for (const f of fingersR) finger(handR, f.map(liftR), 1, 2);
    finger(handR, thumbR.map(liftR), 1, 3);
    const pw = norm(cross(SR, dp));
    pick.path([liftR(add(grip, mul(pw, 1.1))), liftR(sub(grip, mul(pw, 1.1))), liftR(pickTip)], true);
    const wristR = PG(liftR(right.W));

    // ---- the left hand: a chord shape at a fret from the pitch, held until the pitch has moved
    // on by most of a fret, the fingers easing between shapes
    const Ft = 1 + 7 * clamp01(pose.fret);
    if (Math.abs(Ft - F) > 0.75) F = Math.max(1, Math.min(8, Math.round(Ft)));
    const shape = SHAPES[F % SHAPES.length];
    const r = 1 - Math.exp(-dt / 0.07), rh = 1 - Math.exp(-dt / 0.1);
    aHand = lerp(aHand, pressAt(F), rh);
    for (let i = 0; i < 4; i++) {
      const p = shape.fingers[i];
      const target: V3 = i === 0 && shape.barre ? [pressAt(F), 3.0, STRING_C + 0.15] : p ? [pressAt(F + p.f), stringB(p.s, pressAt(F + p.f)), STRING_C] : [aHand - 1.9 * i, -0.6 + 0.3 * i, 7.6];
      tips[i] = ease(tips[i], target, r);
    }
    thumbTip = ease(thumbTip, [aHand + 1.5, 1.5, 0.8], rh);
    // the palm under the neck, the knuckles below its near edge and in front of the strings, the
    // thumb up the back of the neck
    const WL: V3 = [aHand - 4, -10.5, 6.0];
    const KcL: V3 = [aHand - 2.8, -4.6, 8.6];
    const hL = norm(sub(KcL, WL));
    const NL = perp([0, -0.35, 1], hL);
    const SL = norm(cross(hL, NL));
    const left: HandFrame = { W: WL, h: hL, N: NL, S: SL };
    const handL = new Lines();
    palm(handL, left, KcL, 1);
    for (let i = 0; i < 4; i++) finger(handL, reaching(lin([KcL, 1], [SL, KNUCKLE[i]], [hL, i === 3 ? -0.5 : 0]), tips[i], FINGER[i], fingerBend, 1), 1, 3);
    const thumbBaseL: V3 = [aHand - 0.5, -6.0, 2.6];
    finger(handL, [thumbBaseL, ik2(thumbBaseL, thumbTip, THUMB[0], THUMB[1], [-0.2, 0, -1]), thumbTip], 1, 3);
    const wristL = PG(WL);

    // ---- the arms: from the shoulders to the wrists, the elbows bent down and out
    const arms = new Lines();
    const SR3 = shoulder(1), SL3 = shoulder(-1);
    const rest = PG(REST);
    const elbowR = ik2(SR3, rest, 30 * k, 13 * k, [-0.6, -0.2, 1]);
    const pathR = spline([SR3, elbowR, rest, wristR], 4);
    tube(arms, pathR, armR(pathR.length), 8, 4, 16);
    const elbowL = ik2(SL3, wristL, 30 * k, 27 * k, [-0.5, -1, -0.2]);
    const pathL = spline([SL3, elbowL, wristL], 5);
    tube(arms, pathL, armR(pathL.length), 8, 5, 16);

    // ---- the strings, shivering after a strum (a standing wave, aliased to a visible flutter)
    const strings = new Lines();
    const amp = pose.pluck < 1.5 ? 0.45 * Math.exp(-pose.pluck / 0.35) : 0;
    for (let s = 0; s < 6; s++) {
      const pts: V3[] = [];
      const ph = Math.sin(clock * (54 + 7 * s) + s);
      for (let i = 0; i <= 9; i++) {
        const t = i / 9, a = lerp(BRIDGE, NUT, t);
        const w = amp * ph * Math.sin(Math.PI * t) * (0.7 + 0.3 * (s / 5));
        pts.push(PG([a, stringB(s, a) + 0.4 * w, STRING_C + w]));
      }
      strings.path(pts);
    }

    // ---- the right toe tapping
    const foot = new Lines();
    shoe(foot, 1, 0.26 * clamp01(pose.tap));

    const guitarLines = shift(g.lines);
    return [
      { lines: shear(torso.lines), color: V.blueMid, width: 0.85, alpha: 0.55 },
      { lines: [...shear(legs.lines), ...foot.lines], color: V.blueMid, width: 0.85, alpha: 0.5 },
      { lines: arms.lines, color: V.blueMid, width: 0.85, alpha: 0.6 },
      { lines: [...toBody(handR.lines), ...toBody(handL.lines)], color: V.blue, width: 0.85, alpha: 0.7 },
      { lines: toBody(pick.lines), color: V.amber, width: 1.05, alpha: 0.95 },
      { lines: guitarLines, color: V.amber, width: 1.05, alpha: 0.9 },
      { lines: shift(frets.lines), color: V.amber, width: 0.85, alpha: 0.6 },
      { lines: [...strings.lines, ...shift(tails.lines)], color: V.white, width: 0.6, alpha: 0.8 + 0.2 * Math.min(1, amp * 2) },
    ];
  };
  return { scene, frame };
}

/** the props for a body: `face` is its face point (body cm), which a stage holds still */
export function buildProps(head: HeadSections, k: number, face: V3): Props {
  const p = performer(head, k, face);
  return { hat: hat(head, k), scene: p.scene, frame: p.frame };
}
