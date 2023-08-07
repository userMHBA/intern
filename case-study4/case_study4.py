import datascience_full_cheetsheet.feature_engineering as fe
import datascience_full_cheetsheet.kesifci_veri_analizi as kva
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)


def load_data():
    data = pd.read_csv("./diabetes.csv")
    return data


df = load_data()
df.columns = df.columns.str.upper()
df.head()


kva.check_df(df)

cat_cols, num_cols, cat_but_car = kva.grab_col_names(df)

for col in num_cols:
    print("###############################################")
    kva.num_summary(df, col, plot=True)
    print("###############################################")


kva.target_summary_with_num(df, "OUTCOME", num_cols)