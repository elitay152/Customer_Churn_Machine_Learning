import unittest
import numpy as np
import pandas as pd
from churn_workflow import CATEGORICAL, NUMERIC, evaluate_model, model_spec, split_data

class FixedProbabilities:
    def __init__(self, values): self.values = np.asarray(values)
    def predict_proba(self, X): return np.column_stack([1-self.values, self.values])

class WorkflowTests(unittest.TestCase):
    def test_evaluation_uses_probabilities_and_supplied_model(self):
        y = np.array([0, 1, 0, 1])
        good = FixedProbabilities([0.1, 0.4, 0.2, 0.3])
        bad = FixedProbabilities([0.4, 0.1, 0.3, 0.2])
        metrics, _, _ = evaluate_model(good, np.zeros((4, 1)), y)
        self.assertEqual(metrics['roc_auc'], 1.0)
        self.assertEqual(metrics['recall'], 0.0)
        other, _, _ = evaluate_model(bad, np.zeros((4, 1)), y)
        self.assertEqual(other['roc_auc'], 0.0)

    def test_preprocessing_fits_training_only_and_handles_unknowns(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(100, len(NUMERIC))), columns=NUMERIC)
        X['Geography'] = np.tile(['France', 'Spain'], 50)
        X['Gender'] = np.tile(['Female', 'Male'], 50)
        y = pd.Series(np.tile([0, 1], 50))
        train, test, labels, _ = split_data(X, y)
        self.assertFalse(set(train.index) & set(test.index))
        pipeline, _ = model_spec('logistic')
        pipeline.fit(train, labels)
        scaler = pipeline.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
        np.testing.assert_allclose(scaler.mean_, train[NUMERIC].mean())
        self.assertFalse(np.allclose(scaler.mean_, X[NUMERIC].mean()))
        test = test.copy()
        test[CATEGORICAL] = 'previously unseen'
        test.loc[:, NUMERIC] = np.nan
        self.assertTrue(np.isfinite(pipeline.predict_proba(test)).all())

if __name__ == '__main__': unittest.main()
