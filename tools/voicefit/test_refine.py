"""Run with the voicefit environment: python -m unittest discover -s tools/voicefit."""
import unittest

import numpy as np

from refine import apply_vowel_audit, recover_short_vowels, recover_vowel_onsets


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


class ShortVowelTests(unittest.TestCase):
    def track(self, phones):
        return {"phrases": [{"text": "test", "words": [{"w": "test", "phones": phones}]}]}

    def reference(self, intervals):
        def cost(phone, start, end):
            values = [0.1 if any(p == phone and a <= i / 100 < b for p, a, b in intervals) else 2.0
                      for i in range(start, end)]
            return sum(values), len(values)
        return cost

    def test_stressed_vowel_recovers_time_across_a_consonant_cluster(self):
        track = self.track([["M", 0, 0.03], ["AW1", 0.03, 0.06], ["N", 0.06, 0.09],
                            ["T", 0.09, 0.12], ["AH0", 0.12, 0.43], ["N", 0.43, 0.56]])
        cost = self.reference([("AW1", 0.03, 0.29), ("AH0", 0.35, 0.43)])
        self.assertEqual(len(recover_short_vowels(track, cost)), 1)
        ps = track["phrases"][0]["words"][0]["phones"]
        self.assertAlmostEqual(ps[1][2] - ps[1][1], 0.26)
        self.assertEqual((ps[0][1], ps[-1][2]), (0, 0.56))
        self.assertAlmostEqual(ps[2][2] - ps[2][1], 0.03)
        self.assertAlmostEqual(ps[3][2] - ps[3][1], 0.03)
        self.assertEqual(recover_short_vowels(track, cost), [])

    def test_single_vowel_recovers_voiced_time_from_overlong_consonants(self):
        track = self.track([["HH", 0, 0.346], ["OW1", 0.346, 0.383], ["M", 0.383, 0.57]])
        self.assertEqual(len(recover_short_vowels(track, self.reference([("OW1", 0.06, 0.50)]))), 1)
        ps = track["phrases"][0]["words"][0]["phones"]
        self.assertGreater(ps[1][2] - ps[1][1], 0.4)
        self.assertEqual((ps[0][1], ps[-1][2]), (0, 0.57))
        self.assertGreaterEqual(ps[0][2], 0.06)
        self.assertGreaterEqual(ps[-1][2] - ps[-1][1], 0.06 - 1e-9)

    def test_short_words_and_missing_or_unhelpful_evidence_stay_unchanged(self):
        short = self.track([["M", 0, 0.04], ["IY1", 0.04, 0.11]])
        self.assertEqual(recover_short_vowels(short, self.reference([("IY1", 0, 1)])), [])
        for cost in [lambda p,a,b: (0, 0), lambda p,a,b: (b-a, b-a)]:
            ambiguous = self.track([["AW1", 0, 0.03], ["N", 0.03, 0.06], ["AH0", 0.06, 0.5]])
            self.assertEqual(recover_short_vowels(ambiguous, cost), [])

    def test_audit_applies_only_accepted_repairs_and_can_be_retried(self):
        track = self.track([["HH", 0, 0.346], ["OW1", 0.346, 0.383], ["M", 0.383, 0.57]])
        report = {"repairs": [{"line": 0, "word_index": 0, "word": "test", "accepted": False,
                              "before": track["phrases"][0]["words"][0]["phones"],
                              "after": [["HH", 0, 0.06], ["OW1", 0.06, 0.5], ["M", 0.5, 0.57]]}]}
        self.assertEqual(apply_vowel_audit(track, report), 0)
        report["repairs"][0]["accepted"] = True
        self.assertEqual(apply_vowel_audit(track, report), 1)
        self.assertEqual(apply_vowel_audit(track, report), 0)
        track["phrases"][0]["words"][0]["phones"][0][2] = 0.05
        with self.assertRaisesRegex(ValueError, "stale"):
            apply_vowel_audit(track, report)
