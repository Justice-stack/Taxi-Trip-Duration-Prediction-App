import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def predict(model, df):
    y_pred = model.predict(df)

    return y_pred