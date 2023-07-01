# Car Fuel Consumption prediction
:::

::: {.cell .markdown id="uDjacDmrnyD_"}
## Introduction

In this exercise, we\'ll train a regression tree to predict the mpg
(miles per gallon) consumption of cars in the auto-mpg dataset using all
the six available features.
:::

::: {.cell .markdown id="MtaPjkwcoEEx"}
![car](vertopal_d1121d14e9df43f8bb0e91e6c675566e/baeece4ad767f76b67db739e29b451cb92e478f6.jpg)
:::

::: {.cell .code id="XIyN93rAoPI6"}
``` python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import cross_val_score
```
:::

::: {.cell .markdown id="tgdgGx1GoSKB"}
## Dataset
:::

::: {.cell .code id="0xyggkkCoRo7"}
``` python
df_car = pd.read_csv('auto.csv')
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="E2EoQwFkoXmg" outputId="d3c2763c-ba10-482a-9b91-9542a461d7c7"}
``` python
print(df_car.head())
```

::: {.output .stream .stdout}
        mpg  displ   hp  weight  accel  origin  size
    0  18.0  250.0   88    3139   14.5      US  15.0
    1   9.0  304.0  193    4732   18.5      US  20.0
    2  36.1   91.0   60    1800   16.4    Asia  10.0
    3  18.5  250.0   98    3525   19.0      US  15.0
    4  34.3   97.0   78    2188   15.8  Europe  10.0
:::
:::

::: {.cell .code id="yZo6uLM_poGx"}
``` python
# Create dummy variables for Origin column
origin_dummy = pd.get_dummies(df_car['origin'])
df_car = pd.concat([df_car, origin_dummy], axis=1)
df_car = df_car.drop(['origin'], axis=1)
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="yEbhfxMup3sL" outputId="8c2746fa-5375-4ee2-aba0-0c4f62ae3d6f"}
``` python
print(df_car.head())
```

::: {.output .stream .stdout}
        mpg  displ   hp  weight  accel  size  Asia  Europe  US
    0  18.0  250.0   88    3139   14.5  15.0     0       0   1
    1   9.0  304.0  193    4732   18.5  20.0     0       0   1
    2  36.1   91.0   60    1800   16.4  10.0     1       0   0
    3  18.5  250.0   98    3525   19.0  15.0     0       0   1
    4  34.3   97.0   78    2188   15.8  10.0     0       1   0
:::
:::

::: {.cell .code id="6IIG-AZEoZ-L"}
``` python
X = df_car.drop(['mpg'], axis = 1)
y = df_car[['mpg']]
```
:::

::: {.cell .markdown id="6qDrdM2No_c1"}
## Train/Test split
:::

::: {.cell .code id="zua3cK8dokDQ"}
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```
:::

::: {.cell .markdown id="Uh0oKWqapDDU"}
## Train and fit model
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":75}" id="tCuA5B3Ho7Xf" outputId="07e7b177-de4b-413e-b75c-965292292adf"}
``` python
# Instantiate model
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=3)

# Fit to training data
dt.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="31"}
```{=html}
<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeRegressor</label><div class="sk-toggleable__content"><pre>DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=3)</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .markdown id="ySLaH_5fwfcR"}
## Evaluation
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="MgfdxIZdpSPu" outputId="68a1c461-c899-497d-963a-e2ab9eaafcbf"}
``` python
# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse
mse_dt = MSE(y_test, y_pred)

# Compute rmse
rmse_dt = mse_dt ** (1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
```

::: {.output .stream .stdout}
    Test set RMSE of dt: 4.56
:::
:::

::: {.cell .markdown id="wFtVmCoqyOze"}
## Evaluate 10 fold CV error
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="WKCoe_9nwvyw" outputId="5926049a-1668-44d5-a812-b168f0d5091c"}
``` python
# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, scoring = 'neg_mean_squared_error', n_jobs=-1)

# Compute the 1--folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean()) ** (1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))
```

::: {.output .stream .stdout}
    CV RMSE: 4.90
:::
:::

::: {.cell .code id="QMKD2bWkyrO7"}
``` python
```
:::

::: {.cell .code id="ec58JkyhzJMc"}
``` python
```
:::
