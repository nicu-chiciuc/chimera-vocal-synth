/**
 * Voice profiles: a particular speaker on top of a body. A profile is one JSON file with
 *
 * - `body`: the blend and the build (`size`, `pharynx`: multipliers of the blend's anatomy)
 * - `source`: the performer's source settings (the Voice and Tone knobs)
 * - `posture`: rest positions of articulators the plans leave alone (epilarynx, ventricular, velum)
 * - `habits`: offsets added to every planned target, and how lax the tongue is (`reduce`)
 *
 * `tools/voicefit/fit.py` writes these from a fit to a recording. Profiles in `local/` are
 * gitignored: fits to recordings we have no rights to, or of people who did not ask for one,
 * stay on the machine that made them.
 */
import type { BodyMix, Habits, HumanArtic, VoiceSource } from "../wasm/types.ts";

export interface VoiceProfile {
  /** the file's name */
  id: string;
  name: string;
  /** where it came from (a fit's scores, the recording) */
  note?: string;
  body: BodyMix;
  source: Partial<VoiceSource>;
  posture: Partial<HumanArtic>;
  habits: Habits;
}

const files = import.meta.glob<Omit<VoiceProfile, "id">>(["./*.json", "./local/*.json"], { eager: true, import: "default" });

export const VOICES: readonly VoiceProfile[] = Object.entries(files)
  .map(([path, v]) => ({ ...v, id: path.replace(/^.*\/|\.json$/g, "") }))
  .sort((a, b) => a.name.localeCompare(b.name));

export const voiceById = (id: string | null): VoiceProfile | undefined => VOICES.find((v) => v.id === id);
