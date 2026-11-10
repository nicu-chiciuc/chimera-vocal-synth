//! A speaker's habits move every scheduled target as that speaker would, and no habits change
//! nothing.
use chimera::anatomy::human::{ArticKey, BodyMix, NEUTRAL};
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::engine::Engine;
use chimera::phonetics::Segment;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::habits::Habits;

fn render(habits: Option<Habits>) -> Vec<f32> {
    let mut e = Engine::new(48000.0);
    if let Some(h) = habits {
        e.set_habits(h);
    }
    let segs: Vec<Segment> = ["h", "ɛ", "l", "oʊ"].iter().map(|s| Segment::new(s)).collect();
    let plan = plan_utterance(&segs, &PlanOptions { body: BodyMix::MALE, t0: 0.05, ..Default::default() });
    e.schedule(&plan.gestures, 0.0, true);
    let mut out = vec![0f32; 48000];
    for b in out.chunks_mut(128) {
        e.render(b);
    }
    out
}

#[test]
fn no_habits_change_nothing() {
    let a = render(None);
    let b = render(Some(Habits::NONE));
    assert!(a == b, "Habits::NONE must leave the voice bit-identical");
    let c = render(Some(serde_json::from_str("{}").unwrap()));
    assert!(a == c);
}

#[test]
fn habits_move_planned_targets() {
    let h: Habits = serde_json::from_str(r#"{"offset":{"larynx":-0.5,"jaw":0.1},"reduce":0.5}"#).unwrap();
    let mut g = Gesture::new(0.0, 0.1, 0.02, 0.02, 1.0, Targets::default().with(ArticKey::Larynx, 0.3).with(ArticKey::TbX, 4.0).with(ArticKey::LipAp, 1.0));
    h.apply(&mut g);
    assert!((g.targets.get(ArticKey::Larynx).unwrap() - (-0.2)).abs() < 1e-12);
    assert!((g.targets.get(ArticKey::TbX).unwrap() - (4.0 + 0.5 * (NEUTRAL.tb_x - 4.0))).abs() < 1e-12);
    assert_eq!(g.targets.get(ArticKey::LipAp), Some(1.0), "untouched");
    assert!(!g.targets.has(ArticKey::Jaw), "an offset adds to a target, it doesn't create one");
    assert!(!g.targets.has(ControlKey::F0));
    // and they reach the sound
    assert!(render(Some(h)) != render(None));
}

#[test]
fn habits_reject_unknown_articulators() {
    assert!(serde_json::from_str::<Habits>(r#"{"offset":{"elbow":1}}"#).is_err());
}

#[test]
fn habits_survive_a_body_switch() {
    let h: Habits = serde_json::from_str(r#"{"offset":{"larynx":-0.5}}"#).unwrap();
    let mut e = Engine::new(48000.0);
    e.set_habits(h);
    e.set_body(BodyMix::FEMALE);
    assert_eq!(e.voice().habits, h);
}
