from libraries import pd, np, plt


class Eda:
    data_file = 'dataset/Admission_Prediction.csv'

    def __init__(self):
        self.data = None

    def get_file(self):
        dataf = pd.read_csv(self.data_file)
        return dataf

    def main(self):
        data = self.get_file()


