//! `chimera render track <track.json> <out>`: sing a real recording's performance. The track gives
//! the singer's phones with their aligned times (ARPAbet, as the Montreal Forced Aligner returns
//! them), their pitch curve, and optionally a breath-pressure curve; the synth sings those phones
//! on that timeline at that pitch. The voice itself (body size, source settings) comes from the
//! track's `anatomy` and `source`, which is what `tools/voicefit/fit.py` searches.
//!
//! ```json
//! { "body": "male", "duration": 21.0,
//!   "anatomy": { "scale": 1.0, "pharynx": 1.0 },
//!   "source": { "tension": 0.5, "breathiness": 0.2, "fry": 0.1, "roughness": 0.2, "pressure": 8 },
//!   "phrases": [ { "words": [ { "phones": [ ["HH", 0.52, 0.60], ["ER1", 0.60, 0.80], ["T", 0.80, 0.86] ] } ] } ],
//!   "f0": { "hop": 0.005, "hz": [ 0, 98.2, 98.5 ] },
//!   "pressure": { "hop": 0.005, "values": [ 8.0, 8.2 ] },
//!   "articulation": { "offset": { "larynx": -0.4 }, "posture": { "epilarynx": 0.3 }, "reduce": 0.2 } }
//! ```
//!
//! Pitch 0 (unvoiced) holds the last pitch. The planner's own pitch gestures are dropped, so the
//! curve alone sets the pitch.

use crate::js::fixed;
use crate::{Ctx, FS, body, peak, wav_name};
use chimera::anatomy::human::{ArticKey, ArticPatch, NEUTRAL};
use chimera::control::score::ControlKey;
use chimera::phonetics::english::arpabet_to_segments;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::{Intonation, Segment};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde::Deserialize;

#[derive(Deserialize)]
struct Track {
    #[serde(default = "male")]
    body: String,
    duration: f64,
    #[serde(default)]
    anatomy: Anatomy,
    #[serde(default)]
    source: SourcePatch,
    phrases: Vec<Phrase>,
    f0: Curve,
    #[serde(default)]
    pressure: Option<Curve>,
    #[serde(default)]
    articulation: Articulation,
}

/// The singer's habitual setting: `offset` is added to every planned target of an articulator
/// (a lower larynx, a wider jaw, more lip protrusion throughout), `posture` is the rest position
/// of articulators the plans leave alone (epilarynx, ventricular folds, velum). Keys are the
/// engine's articulator names (`larynx`, `jaw`, `lipProt`, ...).
#[derive(Deserialize, Default)]
struct Articulation {
    #[serde(default)]
    offset: std::collections::BTreeMap<String, f64>,
    #[serde(default)]
    posture: std::collections::BTreeMap<String, f64>,
    /// 0..1: how far every tongue-body target moves toward the neutral tongue (lax, centralised
    /// articulation: front vowels' F2 falls, open vowels' F1 falls)
    #[serde(default)]
    reduce: f64,
}

/// The articulators `reduce` pulls toward neutral.
const REDUCED: [ArticKey; 2] = [ArticKey::TbX, ArticKey::TbY];

fn artic_key(name: &str) -> Result<ArticKey, String> {
    ArticKey::from_name(name).ok_or_else(|| format!("unknown articulator {name:?}"))
}

fn male() -> String {
    "male".into()
}

#[derive(Deserialize, Default)]
struct Anatomy {
    scale: Option<f64>,
    pharynx: Option<f64>,
}

#[derive(Deserialize)]
struct Phrase {
    words: Vec<Word>,
}

#[derive(Deserialize)]
struct Word {
    /// (ARPAbet phone, start s, end s)
    phones: Vec<(String, f64, f64)>,
}

#[derive(Deserialize)]
struct Curve {
    hop: f64,
    #[serde(alias = "hz")]
    values: Vec<f64>,
}

impl Curve {
    fn at(&self, t: f64) -> f64 {
        let i = (t / self.hop).floor().max(0.0) as usize;
        self.values
            .get(i.min(self.values.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
    }
}

/// A pause at least this long inside a phrase gets its own breath group: the planner lays a
/// group's segments end to end, so a pause inside one would pull everything after it early.
const SPLIT_GAP: f64 = 0.15;

/// A breath group: its aligned start and end, and its segments, each timed as aligned (a shorter
/// pause is added to the phone before it).
struct Chunk {
    t0: f64,
    t1: f64,
    segs: Vec<Segment>,
}

fn chunks(p: &Phrase) -> Result<Vec<Chunk>, String> {
    let phones: Vec<(&str, f64, f64, bool)> = p
        .words
        .iter()
        .flat_map(|w| {
            w.phones
                .iter()
                .enumerate()
                .map(|(k, x)| (x.0.as_str(), x.1, x.2, k + 1 == w.phones.len()))
        })
        .collect();
    let mut out: Vec<Chunk> = Vec::new();
    for (i, &(arpa, t0, t1, word_end)) in phones.iter().enumerate() {
        let mut s = arpabet_to_segments(arpa);
        if s.len() != 1 {
            return Err(format!("phone {arpa:?} gives {} segments", s.len()));
        }
        let mut s = s.pop().unwrap();
        let next = phones.get(i + 1).map(|x| x.1);
        let gap = next.map(|n| n - t1).unwrap_or(0.0);
        let end = if gap > 0.0 && gap < SPLIT_GAP {
            next.unwrap()
        } else {
            t1
        };
        s.dur = Some((end - t0).max(0.02));
        s.word_end = Some(word_end);
        match out.last_mut() {
            Some(c) if (t0 - c.t1).abs() < 1e-6 => {
                c.segs.push(s);
                c.t1 = end;
            }
            _ => out.push(Chunk {
                t0,
                t1: end,
                segs: vec![s],
            }),
        }
    }
    Ok(out)
}

pub fn render(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let input = a.first().ok_or("render track <track.json> <out>")?;
    let out_name = wav_name(a.get(1).map(String::as_str).unwrap_or("out/track"));
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let tr: Track = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
    let mut b = body(&tr.body)?;
    if let Some(s) = tr.anatomy.scale {
        b.scale = s;
    }
    if let Some(p) = tr.anatomy.pharynx {
        b.pharynx = p;
    }
    let mut v = HumanVoice::new(FS, b, 2);
    let first_f0 = tr
        .f0
        .values
        .iter()
        .copied()
        .find(|&x| x > 0.0)
        .ok_or("no voiced pitch")?;
    let mut src = tr.source.clone();
    src.f0 = Some(first_f0);
    src.voiced = Some(true);
    src.glide = Some(src.glide.unwrap_or(0.015));
    v.set_source(&src);
    let base_pressure = src.pressure.unwrap_or(8.0);
    if !tr.articulation.posture.is_empty() {
        let mut rest = ArticPatch::default();
        for (k, x) in &tr.articulation.posture {
            rest.set(artic_key(k)?, *x);
        }
        v.set_articulation(&rest, None);
    }
    let offsets: Vec<(ControlKey, f64)> = tr
        .articulation
        .offset
        .iter()
        .map(|(k, x)| artic_key(k).map(|k| (ControlKey::from(k), *x)))
        .collect::<Result<_, _>>()?;

    let mut worst = 0.0f64;
    let mut groups = 0;
    for c in tr
        .phrases
        .iter()
        .map(chunks)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
    {
        groups += 1;
        let start = c.t0;
        let mut plan = plan_utterance(
            &c.segs,
            &PlanOptions {
                body: tr.body.as_str().into(),
                t0: start,
                f0: Some(first_f0),
                intonation: Some(Intonation::Flat),
                sung: Some(true),
                ..Default::default()
            },
        );
        // the planner starts the phrase's first segment where it likes: put it on the aligned time
        let shift = start - plan.segments.first().map(|s| s.t0).unwrap_or(start);
        for g in plan.gestures.iter_mut() {
            g.t0 += shift;
            g.t1 += shift;
            g.targets.remove(ControlKey::F0);
            let r = tr.articulation.reduce;
            if r != 0.0 {
                for k in REDUCED {
                    if let Some(x) = g.targets.get(k) {
                        g.targets.set(k, x + r * (NEUTRAL.get(k) - x));
                    }
                }
            }
            for &(k, dx) in &offsets {
                if let Some(x) = g.targets.get(k) {
                    g.targets.set(k, x + dx);
                }
            }
        }
        if let Some(last) = plan.segments.last() {
            worst = worst.max((last.t1 + shift - c.t1).abs());
        }
        v.schedule(&plan.gestures);
    }

    let n = (tr.duration * FS).round() as usize;
    let mut out = vec![0f32; n];
    let hop = 240; // 5 ms
    let mut last_f0 = first_f0;
    for (i, block) in out.chunks_mut(hop).enumerate() {
        let t = (i * hop) as f64 / FS;
        let f = tr.f0.at(t);
        if f > 0.0 {
            last_f0 = f;
        }
        let mut patch = SourcePatch::f0(last_f0);
        if let Some(pc) = &tr.pressure {
            patch.pressure = Some(pc.at(t));
        } else {
            patch.pressure = Some(base_pressure);
        }
        v.set_source(&patch);
        for sub in block.chunks_mut(128) {
            v.render(sub);
        }
    }
    ctx.write_wav(&out_name, &out)?;
    println!(
        "{out_name}: {} s, peak {}, {} phrases in {groups} breath groups, worst group end off by {} ms",
        fixed(n as f64 / FS, 2),
        fixed(peak(&out), 3),
        tr.phrases.len(),
        fixed(worst * 1000.0, 0)
    );
    Ok(())
}
