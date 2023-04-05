import numpy as np

class NaiveBayes():
    def __init__(self):
        pass

    def prior_prob(self):
        prior = {}
        for outcome in np.unique(self.y):
            outcome_count = sum(self.y == outcome)
            prior[outcome] = outcome_count / len(self.y)
        return prior
    
    def likelihood(self, X):
        features = list(self.columns)
        likelihoods = {}

        for feature in features:
            likelihoods[feature] = {}
            for outcome in np.unique(self.y):
                likelihoods[feature].update({outcome:{}})

        for i in range(len(features)):
            feature = features[i]
            for outcome in np.unique(self.y):
                likelihoods[feature][outcome]['mean'] = X[[j for j, k in enumerate(self.y) if k == outcome]][i].mean()
                likelihoods[feature][outcome]['variance'] = X[[j for j, k in enumerate(self.y) if k == outcome]][i].var()
        return likelihoods

    def fit(self, X, y, columns):
        self.X = X
        self.y = y
        self.columns = columns
        self.classes = np.unique(self.y)
        self.prior = self.prior_prob()
        self.likelihoods = self.likelihood(X)

    def predict(self, X):
        results = []
        X = np.array(X)

        for query in X:
            probs_outcome = {}
        
            for outcome in np.unique(self.y):
                prior = self.prior[outcome]
                likelihood = 1

                for feat, feat_val in zip(self.columns, query):
                    mean = self.likelihoods[feat][outcome]['mean']
                    var = self.likelihoods[feat][outcome]['variance']
                    likelihood *= (1/np.sqrt(2*np.pi*var)) * np.exp(-(feat_val - mean)**2 / (2*var))

                posterior_numerator = (likelihood * prior)
                probs_outcome[outcome] = posterior_numerator

        
            result = max(probs_outcome, key = lambda x: probs_outcome[x])
            results.append(result)

        return np.array(results)
