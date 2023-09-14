import numpy as np
import pickle
from sklearn import decomposition
from sklearn.feature_selection import RFE
# from sklearn.svm import SVR

methods_dict = {
    'FA': decomposition.FactorAnalysis,
    'PCA': decomposition.PCA,
    # 'RFE': lambda n_components: RFE(SVR(kernel="linear"), n_features_to_select=n_components, step=1),
    # 'U': Unsupervised,
    # 'S': Supervised,
    # 'M': Manual
}

class DimensionReducer:
    def __init__(self, n_components, reduction_method='FA'):
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.reduction_model = self.get_reduction_model()

    @classmethod
    def from_file(self, file):
        """Constructor for dimensionality reduction method previously fit and saved

        Args:
            file (string): file path
        """
        self.load(file=file)

    def fit(self, X, y=None):
        X_new = np.reshape(X, (-1, X.shape[-1]))
        if y is None:
            return self.reduction_model.fit(X_new)
        else:
            y_new = np.reshape(y, -1)
            return self.reduction_model.fit(X_new, y_new)

    def transform(self, X):
        X_new = np.reshape(X, (-1, X.shape[-1]))
        X_new = self.reduction_model.transform(X_new)
        X_new = np.reshape(X_new, X.shape[:-1] + (self.n_components,))
        return X_new

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump((self.reduction_model, self.n_components, self.reduction_method), f)

    def load(self, file):
        self.reduction_model, self.n_components, self.reduction_method = pickle.load(open(file, 'rb'))
    
    def get_reduction_model(self):
        try:
            return methods_dict[self.reduction_method](self.n_components)
        except:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")