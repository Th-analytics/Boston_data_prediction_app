from libraries import pd, np, plt, pickle, StandardScaler, sklearn
from Traning_Model import Traning_Testing

class Predict:

    def __init__(self):
        self.file_location = 'Model_File/Boston_data.pickle'


    def get_data(self, file_=None):

        model_file = pickle.load(open(file_, 'rb'))
        return model_file

    def check_data(self, dataset):
        try:
            if type(dataset) != list:
                print("Send data in List only format.")
                return False
            else:
                for values in dataset:
                    if type (values) == str:
                        print("Enter a valid format of data")
                        return False
        except Exception as e:
            print("Error in check_list of Predict class.")
            return False
        else:
            return True

    def prediction(self, data):
        """
        This function is used for the task of prediction called from main method.
        :param data: Data given by user
        :return: predicted value

        """
        try:
            scaler = StandardScaler()
            # 'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','BK','LSTAT'

            if self.check_data(data.copy()):
                model = self.get_data(self.file_location)
                transformed_data = self.scaler.transform([data])
                predict = model.predict(transformed_data)
            else:
                return "Error in data "
        except Exception as e:
            predict = None
            print("Error in prediction method of Predict Class:", e)
        else:
            return predict

    def main(self, data):
        try:
            prediction_data = self.prediction(data= data)
        except Exception as e:
            prediction_data =  None
            print("Error in main method of Prediction class:",e)
        else:
            return prediction_data

if __name__ == '__main__':
    obj = Predict()
    obj_t_t = Traning_Testing()
    obj_t_t.main()
    a = obj.main(data = [0.03237, 0.0, 2.18, 0.0, 0.458, 6.998, 45.8, 6.0622, 3.0, 222.0, 18.7, 394.63,2.94])
    print(a)