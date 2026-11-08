/**
 * Stage props for the 3D head (`Anatomy3D.stage`): a wide-brimmed hat on the head, and below it a
 * torso, arms and an acoustic guitar held as a player holds one. Glowing lines like the head.
 *
 * Body cm, the head's frame: x forward, y up, z out of the right side of the face, glottis at the
 * origin. The hat is placed from the head's own cross-sections, so it fits any body; the torso
 * hangs from the bottom of the neck. The hat nods with the head; the torso and guitar don't (the
 * renderer undoes the head's turn for them).
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

export interface Props {
  /** carried by the head (it nods with it) */
  hat: PropGroup[];
  /** standing still under the head */
  body: PropGroup[];
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
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => mul(a, 1 / (len(a) || 1));
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

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

/** a tube along a path: rings at each point, and lines along it */
function tube(L: Lines, path: V3[], radius: (i: number) => number, sides = 6, ringEvery = 1): void {
  const frames = path.map((p, i) => {
    const t = norm(sub(path[Math.min(path.length - 1, i + 1)], path[Math.max(0, i - 1)]));
    const up: V3 = Math.abs(t[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u = norm(cross(up, t));
    return { p, u, v: cross(t, u) };
  });
  const at = (i: number, a: number): V3 => {
    const f = frames[i], r = radius(i);
    return add(f.p, add(mul(f.u, r * Math.cos(a)), mul(f.v, r * Math.sin(a))));
  };
  for (let i = 0; i < path.length; i += ringEvery) {
    for (let s = 0; s <= 16; s++) L.p(at(i, (s / 16) * Math.PI * 2));
    L.end();
  }
  for (let s = 0; s < sides; s++) {
    const a = (s / sides) * Math.PI * 2;
    for (let i = 0; i < path.length; i++) L.p(at(i, a));
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

// ------------------------------------------------------------------ torso, arms, guitar

/** half-width of the guitar's body along its length (cm from the waist, + towards the neck) */
const GUITAR_W = [[-31, 0], [-30.2, 7], [-27.5, 14], [-22, 18.8], [-15, 19.6], [-8, 17], [-2.5, 12.6], [1.5, 12.2], [7, 13.8], [13, 14.6], [18, 12.4], [21.5, 7.5], [22.6, 0]] as const;

function body(head: HeadSections, k: number): PropGroup[] {
  const yn = head.y0 + 1.0 * k;
  const ns = head.at(yn) ?? { F: 3 * k, Bk: -4 * k, W: 5.3 * k };
  const cx = (ns.F + ns.Bk) / 2 - 0.5 * k;
  // the torso: cross-sections down from the base of the neck (y offset, half-width, half-depth)
  const T: [number, number, number][] = [[0, 5.6, 5.0], [-3, 13, 8.5], [-6.5, 19, 10.5], [-12, 18.5, 11.5], [-22, 16.5, 12], [-34, 15.5, 11.5], [-46, 15.8, 11]];
  const at = (i: number, a: number, slope = true): V3 => {
    const [dy, w, d] = T[i];
    const z = w * k * Math.sin(a), x = cx + d * k * Math.cos(a);
    // the shoulders fall away from the neck
    const drop = slope && i <= 2 ? -2.6 * k * (Math.abs(Math.sin(a)) ** 2) * (i / 2) : 0;
    return [x, yn + dy * k + drop, z];
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
  // the guitar's frame: its waist in front of the belly, the neck rising across to the player's
  // left (−z), the strings facing the audience (+x)
  const O: V3 = [cx + 12.5 * k + 5 * k, yn - 30 * k, 3 * k];
  const u = norm([0.05, 0.42, -1]);
  const n = norm([1, 0, 0.12]);
  const v = norm(cross(n, u));
  const P = (a: number, b: number, c = 0): V3 => add(O, add(mul(u, a * k), add(mul(v, b * k), mul(n, c * k))));
  const g = new Lines(), strings = new Lines(), frets = new Lines();
  const outline = (c: number): V3[] => {
    const pts: V3[] = [];
    for (let a = -31; a <= 22.6; a += 1.1) pts.push(P(a, interp(GUITAR_W, a), c));
    const back: V3[] = [];
    for (let a = 22.6; a >= -31; a -= 1.1) back.push(P(a, -interp(GUITAR_W, a), c));
    return [...pts, ...back];
  };
  g.path(outline(5), true);
  g.path(outline(-5), true);
  for (let a = -30; a <= 22; a += 6.5) for (const sd of [1, -1]) g.path([P(a, sd * interp(GUITAR_W, a), 5), P(a, sd * interp(GUITAR_W, a), -5)]);
  // the sound hole, its rosette, the bridge
  for (const r of [4.4, 5.1]) {
    const hole: V3[] = [];
    for (let i = 0; i <= 32; i++) hole.push(P(7 + r * Math.cos((i / 32) * Math.PI * 2), r * Math.sin((i / 32) * Math.PI * 2), 5.05));
    g.path(hole);
  }
  g.path([P(-17, -7.5, 5.1), P(-17, 7.5, 5.1), P(-15, 7.5, 5.1), P(-15, -7.5, 5.1)], true);
  // the neck and fretboard to the nut, then the headstock
  const nut = 66, fb = (a: number) => lerp(2.8, 2.3, (a - 21) / (nut - 21));
  g.path([P(21, -fb(21), 5.4), P(nut, -fb(nut), 5.4), P(nut, fb(nut), 5.4), P(21, fb(21), 5.4)]);
  g.path([P(22.6, -fb(22.6), 1.5), P(nut, -fb(nut), 3.2), P(nut, fb(nut), 3.2), P(22.6, fb(22.6), 1.5)]);
  for (let f = 1; f <= 19; f++) {
    const a = nut - (nut + 16) * (1 - 2 ** (-f / 12));
    if (a < 21) break;
    frets.path([P(a, -fb(a), 5.45), P(a, fb(a), 5.45)]);
  }
  const head3 = [[nut, 2.6], [nut + 4, 4.4], [nut + 17, 4.0], [nut + 18.5, 3.2]] as const;
  g.path([...head3.map(([a, b]) => P(a, b, 4)), ...[...head3].reverse().map(([a, b]) => P(a, -b, 4))], true);
  for (let i = 0; i < 3; i++) for (const sd of [1, -1]) g.path([P(nut + 6 + 4.6 * i, sd * 4.2, 4), P(nut + 6 + 4.6 * i, sd * 6.4, 4)]);
  // six strings, bridge to nut
  for (let s = 0; s < 6; s++) {
    const b0 = -3.2 + (6.4 * s) / 5, b1 = -2.0 + (4.0 * s) / 5;
    strings.path([P(-16, b0, 5.7), P(nut, b1, 5.7), P(nut + 14, b1 * 1.2, 4.2)]);
  }
  // the arms: the right one over the lower bout, its hand at the strings by the sound hole; the
  // left reaching along the neck to the fretboard
  const shoulder = (side: number): V3 => [cx - 0.5 * k, yn - 7.5 * k, side * 18.5 * k];
  const arms = new Lines();
  const right = spline([shoulder(1), [cx - 1 * k, yn - 20 * k, 21 * k], [cx + 2 * k, yn - 30 * k, 21 * k], P(-12, 16, 9), P(-3, 5, 8)], 6);
  const left = spline([shoulder(-1), [cx, yn - 20 * k, -20 * k], [cx + 9 * k, yn - 26 * k, -17 * k], P(36, -10, 7), P(44, -3, 6.5)], 6);
  const r = (n: number) => (i: number) => k * lerp(4.8, 2.8, i / (n - 1));
  tube(arms, right, r(right.length), 8, 4);
  tube(arms, left, r(left.length), 8, 4);
  return [
    { lines: torso.lines, color: V.blueMid, width: 0.85, alpha: 0.55 },
    { lines: arms.lines, color: V.blueMid, width: 0.85, alpha: 0.6 },
    { lines: g.lines, color: V.amber, width: 1.05, alpha: 0.9 },
    { lines: frets.lines, color: V.amber, width: 0.85, alpha: 0.6 },
    { lines: strings.lines, color: V.white, width: 0.6, alpha: 0.8 },
  ];
}

export function buildProps(head: HeadSections, k: number): Props {
  return { hat: hat(head, k), body: body(head, k) };
}
