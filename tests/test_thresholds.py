import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from threshold_analysis import capacity_table, nested_training_scores, threshold_table


class ThresholdTests(unittest.TestCase):
    def test_capacity_ranks_without_exceeding_budget_including_ties(self):
        y = [0, 1, 1, 0, 1]
        table = capacity_table(y, [0.1, 0.9, 0.9, 0.8, 0.2], [0.1, 0.2, 0.4, 1.0])
        self.assertEqual(table.flagged.tolist(), [0, 1, 2, 5])
        self.assertEqual(table.true_positives.tolist(), [0, 1, 2, 3])
        self.assertEqual(table.iloc[1].customers_at_cutoff, 2)
        self.assertAlmostEqual(table.iloc[2].lift_over_random, 1 / 0.6)
        self.assertEqual(table.iloc[-1].recall, 1)
        # Ties are broken by stable input order, not by observing the label.
        tied = capacity_table([0, 1], [0.9, 0.9], [0.5])
        self.assertEqual(tied.iloc[0].true_positives, 0)
        with self.assertRaises(ValueError):
            capacity_table(y, [0.1] * 5, [1.1])

    def test_counts_and_threshold_equality(self):
        table = threshold_table([0, 1, 0, 1], [0.1, 0.5, 0.6, 0.9], [0, 0.5, 1])
        row = table.iloc[1]
        self.assertEqual(row.flagged, 3)
        self.assertEqual(row.true_positives, 2)
        self.assertEqual(row.false_positives, 1)
        self.assertEqual(row.false_negatives, 0)
        self.assertAlmostEqual(row.precision, 2 / 3)
        self.assertEqual(row.contacts_per_1000, 750)
        self.assertEqual(table.iloc[2].precision, 0)
        self.assertTrue(table.flagged.is_monotonic_decreasing)
        self.assertTrue(table.recall.is_monotonic_decreasing)
        self.assertTrue((table.true_positives + table.false_negatives == 2).all())

    def test_invalid_scores_and_thresholds_are_rejected(self):
        for scores in [[np.nan, 0.1], [-0.1, 0.5], [0.2], [0.3, 1.1]]:
            with self.assertRaises(ValueError):
                threshold_table([0, 1], scores)
        with self.assertRaises(ValueError):
            threshold_table([0, 1], [0.1, 0.9], [1.1])

    def test_nested_tuning_never_sees_outer_validation_rows(self):
        X = pd.DataFrame({"value": np.arange(30)}, index=np.arange(100, 130))
        y = pd.Series(np.tile([0, 1], 15), index=X.index)
        scored = []

        def fake_tune(name, train, labels):
            self.assertTrue(train.index.equals(labels.index))
            fitting_indices = set(train.index)

            class FittedSearch:
                def predict_proba(inner_self, validation):
                    self.assertFalse(fitting_indices & set(validation.index))
                    scored.extend(validation.index)
                    return np.tile([0.7, 0.3], (len(validation), 1))

            return FittedSearch()

        with patch("threshold_analysis.tune_model", side_effect=fake_tune) as tune:
            scores = nested_training_scores("logistic", X, y)
        self.assertEqual(tune.call_count, 3)
        self.assertCountEqual(scored, X.index)
        np.testing.assert_allclose(scores, 0.3)


if __name__ == "__main__":
    unittest.main()
