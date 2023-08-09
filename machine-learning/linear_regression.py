######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################


def load():
    data = pd.read_csv("./data/advertising.csv")
    return data


df = load()
df.head()

df.shape

X = df[["TV"]]
y = df["sales"]

############################
# Model
###########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b * w*x

# Sabit (b bias)
reg_model.intercept_


# tv'nin katsayısı (w weight)
reg_model.coef_[0]


####################
# Tahmin
####################

# 150 birimlik tv harcaması olsa ne kadar satış beklenir
reg_model.intercept_ + reg_model.coef_[0] * 150

# 500 birimlik tv harcaması olsa ne kadar satış beklenir
reg_model.intercept_ + reg_model.coef_[0] * 500

# describe
df.describe().T

# veri setindeki maksimum tv 6087 olursa maksimum değer olan 296.40 değeri gelir.
(296.40 - reg_model.intercept_) / reg_model.coef_[0]


########################
# Modeli GörselleştirmeF
########################

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={"color": "b", "s": 9}, ci=False, color="r")

g.set_title(
    f"Model Denklemi: Sales = {round(reg_model.intercept_, 2)} + TV*{round(reg_model.coef_[0], 2)}"
)
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

###################################
# Tahmin Başarısı
###################################

# MSE // 10.51,
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
y.mean()
y.std()

# RMSE / 3.24
np.sqrt(mean_squared_error(y, y_pred))

# MAE // 2.54
mean_absolute_error(y, y_pred)

# R_KARE // 0.61
reg_model.score(X, y)


###################################
# Multilinear Regression
##################################3

df = load()

X = df.drop("sales", axis=1)
y = df[["sales"]]

###################
# Model
###################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)

y_test.shape
y_train.shape

# reg_model = LinearRegression()
# reg_model.fit(X_train, y_train)

reg_model = LinearRegression().fit(X_train, y_train)

# Sabit (b bias)
reg_model.intercept_

# coefficent (w weight)
reg_model.coef_

######################
# Tahmin
######################

# indexleri ile tahmin ediyoruz.
reg_model.intercept_[0] + (
    (reg_model.coef_[0][0] * 30)
    + (reg_model.coef_[0][1] * 10)
    + (reg_model.coef_[0][2] * 40)
)

# Degerleri ile tahmin ediyoruz.
2.90 + 30 * 0.04 + 10 * 0.17 + 40 * 0.002


# yeni gelen verilere göre tahminde bulunuyoruz.
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)  # 6.20


####################
# Tahmin Başarısı
####################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# Train R_KARE
reg_model.score(X_train, y_train)
# 0.89

################################################
# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test R_KARE
reg_model.score(X_test, y_test)
# 0.89

# 10 Katlı CV RMSE
np.mean(
    np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error"))
)
# 1.69

# 5 Katlı CV RMSE
np.mean(
    np.sqrt(-cross_val_score(reg_model, X, y, cv=5, scoring="neg_mean_squared_error"))
)
# 1.71

######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################


# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += y_hat - y
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print(
        "Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(
            initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)
        )
    )

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print(
        "After {0} iterations b = {1}, w = {2}, mse = {3}".format(
            num_iters, b, w, cost_function(Y, b, w, X)
        )
    )
    return cost_history, b, w


df = load()

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
