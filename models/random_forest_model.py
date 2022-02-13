import joblib
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 bootstrap=False, n_jobs=8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split, bootstrap=self.bootstrap,
            n_jobs=n_jobs
        )

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_probabilities(self, x):
        return self.model.predict_proba(x)

    def __call__(self, x, get_probabilities=False):
        if get_probabilities:
            return self.predict_probabilities(x)
        return self.predict(x)

    @staticmethod
    def load_model(path):
        model = RandomForestModel(n_jobs=10)
        model.model = joblib.load(open(path, 'rb'))
        return model
