import joblib
import json
import numpy as np

from azureml.core.model import Model

def int():
    global my_model_1, my_model_2
    model_1_path = Model.get_model_path(model_name='my_first_model')
    model_2_path = Model.get_model_path(model_name='my_second_model')

    model_1 = joblib.load(model_1_path)
    model_2 = joblib.load(model_2_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)

        result_1 = model_1.predict(data)
        result_2 = model_2.predict(data)
        return{"prediction1": result_1.tolist(), "prediction2": result_2.tolist()}
    except Exception as e:
        result = str(e)
        return result   
