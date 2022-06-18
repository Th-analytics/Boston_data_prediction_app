from libraries import pd, np, plt, pickle, StandardScaler, sklearn
from Traning_Model import Traning_Testing

class Predict:

    def __init__(self):
        self.prediction_data = 'Processing...'
        self.file_location = 'Model_File/Boston_data.pickle'


    def get_data(self, file_=None):
        file_ = self.file_location
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

    def get_transform(self):
        try:
            var_scaler = StandardScaler()
            obj_t_t = Traning_Testing()
            var_data = obj_t_t.get_file()
            #self.get_data()
            data_df_ = var_data.reset_index(drop=True).drop('PRICE',axis=1)
            #print(data_df_)
            #print("data_df_.shape",data_df_.shape)
            x_data = var_scaler.fit_transform(data_df_)
            #print("x_data.shape",x_data.shape)
        except Exception as e:
            print("Error in get_transform of Prediction class:",e)
        else:
            return var_scaler


    def prediction(self, data):
        """
        This function is used for the task of prediction called from main method.
        :param data: Data given by user
        :return: predicted value

        """
        try:
            # 'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','BK','LSTAT'

            if self.check_data(data.copy()):
                model = self.get_data(self.file_location)
                #print(data)
                scaler = self.get_transform()
                transformed_data = scaler.transform([data])
                predict = model.predict(transformed_data)
            else:
                return "Error in data "
        except Exception as e:
            predict = None
            print("Error in prediction method of Predict Class:", e)
        else:
            return predict

    def main(self,  data):
        try:
            self.prediction_data = self.prediction(data= data)
            print(self.prediction_data)
        except Exception as e:
            prediction_data = "Error"
            print("Error in main method of Prediction class:",e)


    def result(self):
        return self.prediction_data
"""
if __name__ == '__main__':
    obj = Predict()
    obj_t_t = Traning_Testing()

    #obj_t_t.main()
    #scaler_obj = obj_t_t.scaler
    a = obj.main(data = [0.03237, 0.0, 2.18, 0.0, 0.458, 6.998, 45.8, 6.0622, 3.0, 222.0, 18.7, 394.63,2.94])
    print(a)
    
"""