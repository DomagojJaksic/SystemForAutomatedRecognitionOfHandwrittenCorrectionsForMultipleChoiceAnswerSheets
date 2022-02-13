import joblib
from sklearn.svm import SVC


class SVMModel:

    def __init__(self, kernel='rbf', c=1., gamma='auto', probability=False):
        self.C = c
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.model = SVC(
            kernel=self.kernel, C=self.C, gamma=self.gamma,
            probability=self.probability
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
        model = SVMModel()
        model.model = joblib.load(open(path, 'rb'))
        return model
