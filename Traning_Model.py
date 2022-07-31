from libraries import pd, StandardScaler, train_test_split, LinearRegression, pickle

class Traning_Testing:

    def __init__(self):
        self.data_df = None
        self.scaler = StandardScaler()

    def get_file(self):
        try:
            location = 'dataset/data_boston.csv'
            df_data = pd.read_csv(location)
        except Exception as e:
            print('Error in get_file of Traning_Testing Class:', e)
        else:
            return df_data

    def get_dep_ind_variables(self):
        try:
            data_frame = self.get_file()
            dep_var_d = data_frame['PRICE']
            ind_var_d = data_frame.drop('PRICE', axis=1)
        except Exception as e:
            print("Error in get_dep_ind_variables method of traning_Testing class:", e)
        else:
            return dep_var_d, ind_var_d

    def save_model(self, linear_obj):
        try:
            file_name = 'Model_File/Boston_data.pickle'
            pickle.dump(linear_obj, open(file_name, 'wb'))
        except Exception as e:
            print("Error in save_model of Traning_testing Class")

    def main(self, save='n'):
        dep_var, ind_var = self.get_dep_ind_variables()
        x_data = self.scaler.fit_transform(ind_var)
        x_train, x_test, y_train, y_test = train_test_split(x_data, dep_var, test_size=0.25, random_state=355)
        linear_reg = LinearRegression()
        regression_line = linear_reg.fit(x_train, y_train)
        traning_accuracy = linear_reg.score(x_train, y_train)
        testing_accuracy = linear_reg.score(x_test, y_test)
        print(f"Traning_accuracy:{traning_accuracy} \nTesting_accuracy:{testing_accuracy}")
        # save = input('Save the model(y/n')
        if save.lower() == 'y':
            self.save_model(linear_obj=linear_reg)


"""
if __name__=='__main__':
    obj_t = Traning_Testing()
    obj_t.main(save="y")

"""