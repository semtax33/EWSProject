"""Hard guards separating model research from the observed historical holdout."""

import pandas as pd


def research_view(X, y, cutoff):
    cutoff = pd.Timestamp(cutoff)
    research_X = X.loc[X.index <= cutoff].copy()
    research_y = y.loc[y.index <= cutoff].copy()
    if (research_X.index > cutoff).any() or (research_y.index > cutoff).any():
        raise AssertionError("Research view accessed the historical holdout")
    return research_X, research_y
