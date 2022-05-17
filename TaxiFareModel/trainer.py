import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data, clean_data
class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline():
        """defines the pipeline as a class attribute"""
        """returns a pipelined model"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipeline




    def run(self):
        self.pipeline=Trainer.set_pipeline()


        '''returns a trained pipelined model'''
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def compute_rmse(y_pred, y_true):
        return np.sqrt(((y_pred - y_true)**2).mean())


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = Trainer.compute_rmse(y_pred, y_test)
        #print(rmse)
        return rmse





if __name__ == "__main__":
    # get data
    df=get_data()
    # clean data
    df_clean=clean_data(df, test=False)
    # set X and y
    y = df_clean.pop("fare_amount")
    X = df_clean
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # train
    pipe =Trainer(X_train,y_train)
    pipe.run()
    # evaluate
    evaluation=pipe.evaluate(X_val, y_val)
    print(evaluation)
