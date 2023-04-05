import pandas as pd

class EqualWidthDiscretiser():
    def __init__(self, bins=10, variables=None):
        self.bins = bins
        self.variables = variables

    def fit(self, X):
        self.binner_dict_ = {}

        for var in self.variables:
            tmp, bins = pd.cut(x=X[var], bins=self.bins, retbins=True, duplicates='drop')

            bins = list(bins)
            bins[0] = float("-inf")
            bins[len(bins) - 1] = float("inf")
            self.binner_dict_[var] = bins

        return self

    def transform(self, X):
        for feature in self.variables:
            X[feature] = pd.cut(X[feature], self.binner_dict_[feature], labels=False)

        return X