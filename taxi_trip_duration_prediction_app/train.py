#data preprocessing
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def preprocess_data(df, test_size):

    rs = RobustScaler()

    df_1 = df.copy()
    #split dataframe into target and training set
    X = df_1.copy()
    Y = X.pop("trip_duration_mins")

    #split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state = 41)

    #Scale Features using RobustScaler
    rs.fit(X_train)
    X_train_transformed = rs.transform(X_train)
    X_test_transformed = rs.transform(X_test)

    return X_train_transformed,Y_train, X_test_transformed, Y_test

def train_model(X_train, Y_train):

    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    return lr