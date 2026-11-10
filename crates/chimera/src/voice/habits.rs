//! A speaker's articulatory habits: how this person holds the vocal tract while saying anything.
//! The planners give every speaker of a body the same targets; habits move them the way a
//! particular speaker does throughout (a lowered larynx, a jaw held more closed, lips pushed
//! forward, a lax tongue that undershoots its vowels). They act on gestures as they are
//! scheduled, so speech, song and live notes all get them, without allocating.

use crate::anatomy::human::{ArticKey, N_ARTIC, NEUTRAL};
use crate::control::score::{ControlKey, Gesture};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Habits {
    /// added to every planned target of each articulator (its units: cm, or 0..1)
    pub offset: [f64; N_ARTIC],
    /// 0..1: how far every tongue-body target moves toward the neutral tongue (lax, centralised
    /// articulation: front vowels' F2 falls, open vowels' F1 falls)
    pub reduce: f64,
}

/// The articulators `reduce` pulls toward neutral.
const REDUCED: [ArticKey; 2] = [ArticKey::TbX, ArticKey::TbY];

impl Habits {
    pub const NONE: Habits = Habits { offset: [0.0; N_ARTIC], reduce: 0.0 };

    pub fn is_none(&self) -> bool {
        *self == Self::NONE
    }

    /// Move a planned gesture's articulator targets as this speaker would.
    pub fn apply(&self, g: &mut Gesture) {
        if self.reduce != 0.0 {
            for k in REDUCED {
                if let Some(x) = g.targets.get(k) {
                    g.targets.set(k, x + self.reduce * (NEUTRAL.get(k) - x));
                }
            }
        }
        for (i, &dx) in self.offset.iter().enumerate() {
            if dx != 0.0 {
                let k = ControlKey::Artic(ArticKey::ALL[i]);
                if let Some(x) = g.targets.get(k) {
                    g.targets.set(k, x + dx);
                }
            }
        }
    }
}

impl Default for Habits {
    fn default() -> Self {
        Self::NONE
    }
}

/// `{ "offset": { "larynx": -0.4, "jaw": 0.1 }, "reduce": 0.2 }`, articulators by the engine's
/// names; anything left out is 0.
#[derive(Serialize, Deserialize, Default)]
struct Wire {
    #[serde(default)]
    offset: BTreeMap<String, f64>,
    #[serde(default)]
    reduce: f64,
}

impl<'de> Deserialize<'de> for Habits {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        let mut h = Habits { reduce: w.reduce.clamp(0.0, 1.0), ..Habits::NONE };
        for (k, x) in w.offset {
            let key = ArticKey::from_name(&k).ok_or_else(|| serde::de::Error::custom(format!("unknown articulator {k:?}")))?;
            h.offset[key.index()] = if x.is_finite() { x } else { 0.0 };
        }
        Ok(h)
    }
}

impl Serialize for Habits {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let offset = ArticKey::ALL.iter().zip(self.offset).filter(|(_, x)| *x != 0.0).map(|(k, x)| (k.name().to_string(), x)).collect();
        Wire { offset, reduce: self.reduce }.serialize(s)
    }
}
