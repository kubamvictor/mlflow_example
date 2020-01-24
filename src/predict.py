import os
import pickle
import pandas as pd

project_path = os.getcwd()

# Get test dataset for scoring
test_dataset = pd.read_csv(project_path+"/data/splitter/test.csv",sep=",").iloc[:,0:-1]


def apply(input_dataframe):
    """
    Function to predict price of house base on house features

    Input: 
        input_dataframe: A pandas dataframe with each row representing the features of house 
    Output: 
        output array: An array with the predictions for each row
    """

    # import model
    
    filename = project_path+"/model/housing_model.sav"

    with open(filename,'rb') as model_file:
        pickle_model = pickle.load(model_file)

    return pickle_model.predict(input_dataframe)


# Example
# -----------------------

def scoring_test(input_data, prediction_function):
    """
    Function to test the prediction function with sample data
    """
    results = prediction_function(input_data)

    print (" Prediction was successfull... \n")
    print(" Prediction Results\n")
    print(results)




