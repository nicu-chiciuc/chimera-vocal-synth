"""Run with the voicefit environment: python -m unittest discover -s tools/voicefit."""
import unittest

import numpy as np

from refine import recover_vowel_onsets


class VowelOnsetTests(unittest.TestCase):
    def track(self, phone="AA1", previous_end=0.0):
        phrases = []
        if previous_end:
            phrases.append({"text": "previous", "words": [{"phones": [["IY1", 0.1, previous_end]]}]})
        phrases.append({"text": "all my memories", "words": [
            {"phones": [[phone, 1.236, 1.276], ["L", 1.276, 1.316]]},
            {"phones": [["M", 1.316, 1.35], ["AY1", 1.35, 1.8]]},
        ]})
        return {"phrases": phrases}

    def test_clipped_all_recovers_its_voiced_onset_without_moving_my(self):
        track = self.track()
        voiced = np.zeros(200, dtype=bool)
        voiced[100:180] = True
        db = np.where(voiced, -20.0, -70.0)
        self.assertEqual(recover_vowel_onsets(track, db, voiced), 1)
        phones = track["phrases"][0]["words"]
        self.assertEqual(phones[0]["phones"], [["AA1", 1.0, 1.276], ["L", 1.276, 1.316]])
        self.assertEqual(phones[1]["phones"][0], ["M", 1.316, 1.35])
        self.assertEqual(recover_vowel_onsets(track, db, voiced), 0)

    def test_does_not_cross_a_pause_or_take_a_previous_phrase(self):
        for previous, onset in [(0.0, 1.236), (1.1, 1.1)]:
            track = self.track(previous_end=previous)
            voiced = np.ones(200, dtype=bool)
            if not previous:
                voiced[110:120] = False
            recover_vowel_onsets(track, np.full(200, -20.0), voiced)
            self.assertEqual(track["phrases"][-1]["words"][0]["phones"][0][1], onset)

    def test_does_not_expand_consonants_or_weak_pitched_leakage(self):
        voiced = np.ones(200, dtype=bool)
        db = np.full(200, -60.0)
        db[123:130] = -20.0
        self.assertEqual(recover_vowel_onsets(self.track(), db, voiced), 0)
        self.assertEqual(recover_vowel_onsets(self.track("HH"), np.full(200, -20.0), voiced), 0)
