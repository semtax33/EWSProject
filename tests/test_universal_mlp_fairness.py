import unittest

import numpy as np
import pandas as pd

from run_universal_mlp_fairness import (
    ALLOCATION_POLICY,
    TARGET_MODE,
    UNIVERSAL_FEATURES,
    pooled_walk_forward_predict,
)


class UniversalMLPFairnessTests(unittest.TestCase):
    def _source(self, key, X, y):
        return {
            "key": key,
            "X": X,
            "target": pd.DataFrame({"y": y}, index=X.index),
        }

    def test_protocol_uses_one_exact_specification(self):
        self.assertEqual(TARGET_MODE, "cash_excess")
        self.assertEqual(ALLOCATION_POLICY, "expanding_percentile")
        self.assertEqual(len(UNIVERSAL_FEATURES), 4)
        self.assertEqual(len(set(UNIVERSAL_FEATURES)), 4)

    def test_leave_one_market_out_is_deterministic_and_causal(self):
        index = pd.date_range("2000-01-31", periods=50, freq="ME")
        values = np.arange(len(index), dtype=float)
        X = pd.DataFrame(
            {
                UNIVERSAL_FEATURES[0]: np.sin(values / 4),
                UNIVERSAL_FEATURES[1]: np.cos(values / 7),
                UNIVERSAL_FEATURES[2]: np.sin(values / 9) + values / 100,
                UNIVERSAL_FEATURES[3]: np.cos(values / 5) - values / 120,
            },
            index=index,
        )
        y1 = pd.Series((np.sin(values / 3) > 0).astype(float), index=index)
        y2 = pd.Series((np.cos(values / 4) > 0).astype(float), index=index)
        sources = [self._source("a", X, y1), self._source("b", X, y2)]
        params = {
            "hidden_layer_sizes": (2,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.1,
            "max_iter": 500,
            "shuffle": False,
        }
        kwargs = dict(
            X_test=X,
            source_markets=sources,
            eval_start=index[25],
            eval_end=index[35],
            min_train=12,
            purge=3,
            refit_every=1,
            mlp_params=params,
            random_state=17,
        )
        first = pooled_walk_forward_predict(**kwargs)
        second = pooled_walk_forward_predict(**kwargs)
        pd.testing.assert_series_equal(first, second)
        self.assertTrue(first.notna().all())

        # Labels later than the last prediction's purged cutoff cannot affect it.
        mutated_y1 = y1.copy()
        mutated_y2 = y2.copy()
        cutoff = (index[35].to_period("M") - 3).to_timestamp("M")
        mutated_y1.loc[mutated_y1.index > cutoff] = 1 - mutated_y1.loc[
            mutated_y1.index > cutoff
        ]
        mutated_y2.loc[mutated_y2.index > cutoff] = 1 - mutated_y2.loc[
            mutated_y2.index > cutoff
        ]
        mutated = [
            self._source("a", X, mutated_y1),
            self._source("b", X, mutated_y2),
        ]
        causal = pooled_walk_forward_predict(
            **{**kwargs, "source_markets": mutated}
        )
        pd.testing.assert_series_equal(first, causal)


if __name__ == "__main__":
    unittest.main()
