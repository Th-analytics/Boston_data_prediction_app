from libraries import pd, np, plt, pickle, StandardScaler, sklearn
from sklearn.datasets import load_boston

class Predict:
    model_file = 'Model_File/Admission_prediction.pickle'

    def __init__(self):
        self.model = None

    def get_file(self):
        boston = load_boston()
        return pd.DataFrame(boston.data)

    def main(self):
        self.data =




if __name__ == '__main__':
    obj = Predict()

