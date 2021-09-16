import math
import pickle
import os
import dill
from numpy import put_along_axis
import pandas as pd
import json

class Prediction:

    model = None

    def __init__(self):
        currrentPath = os.path.dirname(os.path.abspath(__file__))
        parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))
        filename = parentPath+'/modelHousesModel/modelHousesModel'
        print(filename)
        infile = open(filename,'rb')
        self.model = pickle.load(infile)
        infile.close()


    def makePrediction(self, inputDF):
        sq_mt_built = []
        precio_m2_barrio = []
        n_rooms = []
        longitude = []
        house_type_id_Pisos = []
        house_type_id_Aticos = []
        house_type_id_Duplex = []
        house_type_id_CasaChalet = []
        is_exterior_0 = []
        is_exterior_1 = []
        has_lift_0 = []
        has_lift_1 = []
        has_pool_0 = []
        has_pool_1 = []
        for index, row in inputDF.iterrows():
            sq_mt_built = row['sq_mt_built']
            precio_m2_barrio = row['precio_m2_barrio']
            n_rooms = row['n_rooms']
            longitude = row['longitude']
            house_type_id_Pisos = 0
            house_type_id_Aticos = 0
            house_type_id_Duplex = 0
            house_type_id_CasaChalet = 0
            is_exterior_0 = 0
            is_exterior_1 = 0
            has_lift_0 = 0
            has_lift_1 = 0
            has_pool_0 = 0
            has_pool_1 = 0

            if row['is_exterior'] == 'TRUE':
                is_exterior_1 = 1

            if row['has_lift'] == 'TRUE':
                has_lift_1 = 1   

            if row['has_pool'] == 'TRUE':
                has_pool_1 = 1        

            if row['house_type_id'] == 'Pisos':
                house_type_id_Pisos = 1
            elif row['house_type_id'] == 'Aticos':
                house_type_id_Aticos = 1
            elif row['house_type_id'] == 'Duplex':
                house_type_id_Duplex = 1
            elif row['house_type_id'] == 'Casa o chalet':
                house_type_id_CasaChalet = 1


        d = {'sq_mt_built' : [sq_mt_built],
             'precio_m2_barrio' : [precio_m2_barrio],
             'n_rooms':[n_rooms],
            'longitude':[longitude],
            'house_type_id_Pisos':[house_type_id_Pisos],
            'house_type_id_Aticos':[house_type_id_Aticos],
            'house_type_id_Duplex':[house_type_id_Duplex],
            'house_type_id_CasaChalet':[house_type_id_CasaChalet],
            'is_exterior_0':[is_exterior_0],
            'is_exterior_1':[is_exterior_1],
            'has_lift_0':[has_lift_0],
            'has_lift_1':[has_lift_1],
            'has_pool_0':[has_pool_0],
            'has_pool_1':[has_pool_1]
             }        
        df = pd.DataFrame(d)
    
        #Transform input DF into proper format
        df = df.values

        #Predict values from input
        predictions = math.e**(self.model.predict(df))
        
        #Transform output prediction into dataframe
        predictionDF = pd.DataFrame(predictions, columns=['prediction'])

        #Build output response object 
        Body = []
        for index, row in inputDF.iterrows():
            sq_mt_built = row['sq_mt_built']
            precio_m2_barrio = row['precio_m2_barrio']
            n_rooms = row['n_rooms']
            longitude = row['longitude']
            house_type_id = row['house_type_id']
            is_exterior = row['is_exterior']
            has_lift = row['has_lift']
            has_pool = row['has_pool']

            prediction = predictionDF.iloc[index]['prediction']

            InputVals = {}

            InputVals['sq_mt_built'] = sq_mt_built
            InputVals['precio_m2_barrio'] = precio_m2_barrio
            InputVals['n_rooms'] = n_rooms
            InputVals['longitude'] = longitude
            InputVals['house_type_id'] = house_type_id
            InputVals['is_exterior'] = is_exterior
            InputVals['has_lift'] = has_lift
            InputVals['has_pool'] = has_pool
            PredictionBody = {}

            PredictionBody['input'] = InputVals
            PredictionBody['prediction'] = prediction

            Body.append(PredictionBody)

            output = json.dumps(Body)
       
        return output


if __name__ == "__main__":

    data2 = [[10.0, 10.0, 3.0, 40.4575, 'Casa o chalet','TRUE','TRUE','TRUE']]
    df2 = pd.DataFrame(data2, columns = [
        'sq_mt_built',
        'precio_m2_barrio',
        'n_rooms',
        'longitude',
        'house_type_id',
        'is_exterior',
        'has_lift',
        'has_pool'

    ])

    predictionObject = Prediction()

    predictionObject.makePrediction(df2)
    currrentPath = os.path.dirname(os.path.abspath(__file__))   
    parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))
    filenameDill = parentPath+'/predictionHousesModel/prediction'
    with open(filenameDill, "wb") as f:
     dill.dump(predictionObject, f)