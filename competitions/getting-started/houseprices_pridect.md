
```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # some plotting!
import seaborn as sns # so pretty!
from scipy import stats # I might use this
from sklearn.ensemble import RandomForestClassifier # checking if this is available
# from sklearn import cross_validation
%matplotlib inline
```


```python
# import the training data set and make sure it's in correctly...(特征初印象)
train = pd.read_csv('C:/Users/1/Documents/House_Prices/train.csv')
train_original = pd.read_csv('C:/Users/1/Documents/House_Prices/train.csv')
test = pd.read_csv('C:/Users/1/Documents/House_Prices/test.csv')
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    


```python
# define a function to convert an object (categorical) feature into an int feature（制定转化类别特征的函数：最常出现的设为0，后续次之）
# 0 = most common category, highest int = least common.
def getObjectFeature(df, col, datalength=1460):
    if df[col].dtype!='object': # if it's not categorical..
        print('feature',col,'is not an object feature.')
        return df
    elif len([i for i in df[col].T.notnull() if i == True])!=datalength: # if there's missing data..
        print('feature',col,'is missing data.')
        return df
    else:
        df1 = df
        counts = df1[col].value_counts() # get the counts for each label for the feature
        df1[col] = [counts.index.tolist().index(i) for i in df1[col]] # do the conversion
        return df1 # make the new (integer) column from the conversion
# and test the function...
fcntest = getObjectFeature(train,'LotShape')
fcntest.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>0</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>0</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>50</td>
      <td>RL</td>
      <td>85.0</td>
      <td>14115</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>10</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>143000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>10084</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>0</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>307000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>10382</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Shed</td>
      <td>350</td>
      <td>11</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>200000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>50</td>
      <td>RM</td>
      <td>51.0</td>
      <td>6120</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>0</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2008</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>129900</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>190</td>
      <td>RL</td>
      <td>50.0</td>
      <td>7420</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>0</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>118000</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 81 columns</p>
</div>




```python
from sklearn.tree import DecisionTreeRegressor as dtr
# define the training data X...（使用决策树回归，只是用四个特征预测）
X = train[['MoSold','YrSold','LotArea','BedroomAbvGr']]
Y = train[['SalePrice']]
# and the data for the competition submission...
X_test = test[['MoSold','YrSold','LotArea','BedroomAbvGr']]
print(X.head())
print(Y.head())
```

       MoSold  YrSold  LotArea  BedroomAbvGr
    0       2    2008     8450             3
    1       5    2007     9600             3
    2       9    2008    11250             3
    3       2    2006     9550             3
    4      12    2008    14260             4
       SalePrice
    0     208500
    1     181500
    2     223500
    3     140000
    4     250000
    


```python
# let's set up some cross-validation analysis to evaluate our model and later models...（使用EV=1−Var(y−y¯)/Var(y)来评价模型，并查看预测的y^和y是否为同分布）
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, make_scorer
explained_variance = make_scorer(explained_variance_score)
# try fitting a decision tree regression model...
DTR_1 = dtr(max_depth=None) # declare the regression model form. Let the depth be default.
# DTR_1.fit(X,Y) # fit the training data
scores_dtr = cross_val_score(DTR_1, X, Y, cv=10, scoring = explained_variance) # 10-fold cross validation
print('scores for k=10 fold validation:',scores_dtr)
print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_dtr.mean(), scores_dtr.std() * 2))
```

    scores for k=10 fold validation: [-0.72005562 -0.50326551 -0.40914348 -0.69462948 -0.21176829 -0.46169978
     -0.77665464 -1.06981755 -0.94466445 -0.3397173 ]
    Est. explained variance: -0.61 (+/- 0.52)
    


```python
from sklearn.ensemble import RandomForestRegressor as rfr
estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []
yt = [i for i in Y['SalePrice']] # quick pre-processing of the target
np.random.seed(11111)
for i in estimators:
    model = rfr(n_estimators=i,max_depth=None)
    scores_rfr = cross_val_score(model,X,yt,cv=10,scoring=explained_variance)
    print('estimators:',i)
    print('explained variance scores for k=10 fold validation:',scores_rfr)
    print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
    print('')
    mean_rfrs.append(scores_rfr.mean())
    std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
    std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting
```

    estimators: 2
    explained variance scores for k=10 fold validation: [-0.22383345 -0.18796622 -0.09511974 -0.18357319 -0.13538499  0.04648584
     -0.59195128 -0.59386179 -0.3091302  -0.28098849]
    Est. explained variance: -0.26 (+/- 0.39)
    
    estimators: 5
    explained variance scores for k=10 fold validation: [ 0.05647738  0.1032377  -0.04072007  0.07757209  0.02874692  0.03047768
     -0.13039353 -0.1470963  -0.09418176 -0.01914147]
    Est. explained variance: -0.01 (+/- 0.17)
    
    estimators: 10
    explained variance scores for k=10 fold validation: [ 0.04485663  0.26456864  0.09660283  0.1198101   0.03766747  0.27337131
     -0.13179273 -0.09630698  0.02393398  0.09364117]
    Est. explained variance: 0.07 (+/- 0.25)
    
    estimators: 15
    explained variance scores for k=10 fold validation: [ 0.04627312  0.28637859  0.05597032  0.16563461  0.17260695  0.21979237
      0.00969847 -0.05699684 -0.00784888  0.13081076]
    Est. explained variance: 0.10 (+/- 0.21)
    
    estimators: 20
    explained variance scores for k=10 fold validation: [ 0.04382324  0.2600127   0.03265876  0.16149146  0.13810078  0.30213336
     -0.02737594  0.03263671 -0.01555347  0.10640119]
    Est. explained variance: 0.10 (+/- 0.21)
    
    estimators: 25
    explained variance scores for k=10 fold validation: [-0.02108627  0.27823741  0.06236606  0.17440568  0.1515791   0.22975024
      0.05558142 -0.04058066 -0.01808912  0.11978249]
    Est. explained variance: 0.10 (+/- 0.21)
    
    estimators: 30
    explained variance scores for k=10 fold validation: [ 0.11099131  0.21410661  0.12834581  0.13008373  0.18660933  0.23288588
      0.00296744 -0.03401822  0.04715847  0.17688066]
    Est. explained variance: 0.12 (+/- 0.17)
    
    estimators: 35
    explained variance scores for k=10 fold validation: [ 0.13624784  0.25096198  0.02025751  0.13387035  0.19969439  0.22663755
     -0.00093639  0.08891239  0.05114151  0.19337615]
    Est. explained variance: 0.13 (+/- 0.17)
    
    estimators: 40
    explained variance scores for k=10 fold validation: [ 0.11772482  0.27259906  0.06937854  0.17174145  0.19962522  0.30475079
      0.01629621  0.02715896 -0.00978731  0.21673874]
    Est. explained variance: 0.14 (+/- 0.21)
    
    estimators: 45
    explained variance scores for k=10 fold validation: [ 0.13534705  0.29851922  0.09171833  0.17385729  0.1841072   0.21630645
      0.0364885   0.03718453  0.01890063  0.16466498]
    Est. explained variance: 0.14 (+/- 0.17)
    
    estimators: 50
    explained variance scores for k=10 fold validation: [ 0.14859327  0.28355526  0.13010835  0.15661145  0.19195547  0.24320294
     -0.04389419  0.04637779  0.01916105  0.11715098]
    Est. explained variance: 0.13 (+/- 0.19)
    
    estimators: 55
    explained variance scores for k=10 fold validation: [ 0.11287017  0.2547342   0.11758528  0.19954875  0.20149773  0.25980275
      0.01365299  0.01155233 -0.00879478  0.15643853]
    Est. explained variance: 0.13 (+/- 0.19)
    
    estimators: 60
    explained variance scores for k=10 fold validation: [ 0.08679155  0.28290124  0.10647687  0.14907707  0.18484815  0.24489571
      0.03074887  0.04967029  0.02456044  0.17669471]
    Est. explained variance: 0.13 (+/- 0.17)
    
    estimators: 65
    explained variance scores for k=10 fold validation: [ 0.11983135  0.30225712  0.09067288  0.13988418  0.18279726  0.2720963
     -0.00403447  0.04807178  0.05619388  0.16443933]
    Est. explained variance: 0.14 (+/- 0.18)
    
    estimators: 70
    explained variance scores for k=10 fold validation: [ 0.10804628  0.28945144  0.11932511  0.18062901  0.21331268  0.2818073
      0.0732552   0.02959393  0.01938409  0.18480725]
    Est. explained variance: 0.15 (+/- 0.18)
    
    estimators: 75
    explained variance scores for k=10 fold validation: [ 0.09095357  0.25641012  0.11236095  0.14465076  0.19616873  0.27212233
      0.04992776  0.06155036  0.05011507  0.18056643]
    Est. explained variance: 0.14 (+/- 0.16)
    
    estimators: 80
    explained variance scores for k=10 fold validation: [ 0.12351231  0.2931243   0.10447381  0.173131    0.19311969  0.25121609
      0.04710564  0.03810432  0.01445248  0.21509248]
    Est. explained variance: 0.15 (+/- 0.18)
    
    


```python
# and plot...
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(estimators,mean_rfrs,marker='o',
       linewidth=4,markersize=12)
ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,
                facecolor='green',alpha=0.3,interpolate=True)
ax.set_ylim([-.3,1])
ax.set_xlim([0,80])
plt.title('Expected Variance of Random Forest Regressor')
plt.ylabel('Expected Variance')
plt.xlabel('Trees in Forest')
plt.grid()
plt.show()
```


![png](output_6_0.png)



```python
# list all the features we want. This is still arbitrary...(添加了后续想要的7个特征)
included_features = ['MoSold','YrSold','LotArea','BedroomAbvGr', # original data
                    'FullBath','HalfBath','TotRmsAbvGrd', # bathrooms and total rooms
                    'YearBuilt','YearRemodAdd', # age of the house
                    'LotShape','Utilities'] # some categoricals 
# define the training data X...
X = train[included_features]
Y = train[['SalePrice']]
# and the data for the competition submission...
X_test = test[included_features]
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype=='object':
        X = getObjectFeature(X, col)
X.head()
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotArea</th>
      <th>BedroomAbvGr</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>TotRmsAbvGrd</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>LotShape</th>
      <th>Utilities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2008</td>
      <td>8450</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>2003</td>
      <td>2003</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>2007</td>
      <td>9600</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2008</td>
      <td>11250</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>2001</td>
      <td>2002</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2006</td>
      <td>9550</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1915</td>
      <td>1970</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>2008</td>
      <td>14260</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>2000</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# define the number of estimators to consider
estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []
yt = [i for i in Y['SalePrice']]
np.random.seed(11111)
# for each number of estimators, fit the model and find the results for 8-fold cross validation
for i in estimators:
    model = rfr(n_estimators=i,max_depth=None)
    scores_rfr = cross_val_score(model,X,yt,cv=10,scoring=explained_variance)
    print('estimators:',i)
    print('explained variance scores for k=10 fold validation:',scores_rfr)
    print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
    print("")
    mean_rfrs.append(scores_rfr.mean())
    std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
    std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting
```

    estimators: 2
    explained variance scores for k=10 fold validation: [ 0.4758688   0.5695577   0.67719218  0.50939472  0.69103923  0.65177368
      0.59343988  0.57860855  0.40682187  0.51496108]
    Est. explained variance: 0.57 (+/- 0.17)
    
    estimators: 5
    explained variance scores for k=10 fold validation: [ 0.4927927   0.59539611  0.69793342  0.62247622  0.76854154  0.80695392
      0.65484316  0.62402942  0.41767281  0.6710954 ]
    Est. explained variance: 0.64 (+/- 0.22)
    
    estimators: 10
    explained variance scores for k=10 fold validation: [ 0.51323372  0.65614554  0.72441015  0.59984307  0.77862197  0.76227133
      0.67464334  0.65810923  0.42029668  0.5390159 ]
    Est. explained variance: 0.63 (+/- 0.22)
    
    estimators: 15
    explained variance scores for k=10 fold validation: [ 0.59727784  0.65921834  0.73048545  0.60506536  0.76715971  0.80314833
      0.71476741  0.71562418  0.53769993  0.65801362]
    Est. explained variance: 0.68 (+/- 0.16)
    
    estimators: 20
    explained variance scores for k=10 fold validation: [ 0.63917871  0.68159311  0.77934901  0.6444845   0.77509668  0.78661283
      0.66709014  0.71951881  0.4986433   0.62103898]
    Est. explained variance: 0.68 (+/- 0.17)
    
    estimators: 25
    explained variance scores for k=10 fold validation: [ 0.60115645  0.66598128  0.76107826  0.63252061  0.77216564  0.75015763
      0.7037534   0.73807907  0.40042338  0.62983117]
    Est. explained variance: 0.67 (+/- 0.21)
    
    estimators: 30
    explained variance scores for k=10 fold validation: [ 0.58113416  0.69263108  0.75611471  0.6442029   0.77331231  0.78656981
      0.6911571   0.70019526  0.5341112   0.61678092]
    Est. explained variance: 0.68 (+/- 0.16)
    
    estimators: 35
    explained variance scores for k=10 fold validation: [ 0.60056562  0.68377346  0.76279583  0.64980323  0.78494817  0.83284115
      0.7016248   0.70502337  0.42499253  0.6523261 ]
    Est. explained variance: 0.68 (+/- 0.21)
    
    estimators: 40
    explained variance scores for k=10 fold validation: [ 0.60748651  0.69704104  0.7646751   0.64934611  0.78452011  0.78849942
      0.69777375  0.74486298  0.40759774  0.63104072]
    Est. explained variance: 0.68 (+/- 0.22)
    
    estimators: 45
    explained variance scores for k=10 fold validation: [ 0.59355084  0.69132821  0.7565988   0.66676984  0.78953642  0.76894267
      0.69696846  0.72886695  0.42043414  0.64214401]
    Est. explained variance: 0.68 (+/- 0.20)
    
    estimators: 50
    explained variance scores for k=10 fold validation: [ 0.65158986  0.69181654  0.75797462  0.65278916  0.78792601  0.82599999
      0.70092089  0.73766645  0.4415123   0.64198463]
    Est. explained variance: 0.69 (+/- 0.20)
    
    estimators: 55
    explained variance scores for k=10 fold validation: [ 0.59589868  0.68656687  0.77645982  0.64037516  0.78808846  0.78987985
      0.70119442  0.73424577  0.41451852  0.61202763]
    Est. explained variance: 0.67 (+/- 0.22)
    
    estimators: 60
    explained variance scores for k=10 fold validation: [ 0.62689993  0.70013828  0.7612831   0.65821566  0.78405006  0.80507777
      0.70510205  0.7411017   0.47495258  0.61521427]
    Est. explained variance: 0.69 (+/- 0.19)
    
    estimators: 65
    explained variance scores for k=10 fold validation: [ 0.63557279  0.68915126  0.75130194  0.65320677  0.78745138  0.79769963
      0.70054063  0.75759699  0.45462999  0.62959014]
    Est. explained variance: 0.69 (+/- 0.19)
    
    estimators: 70
    explained variance scores for k=10 fold validation: [ 0.63254447  0.67873379  0.76011035  0.66286341  0.78369339  0.80585469
      0.69844871  0.72135359  0.47664602  0.62230633]
    Est. explained variance: 0.68 (+/- 0.18)
    
    estimators: 75
    explained variance scores for k=10 fold validation: [ 0.63355641  0.68964301  0.75008594  0.66244164  0.79712277  0.81213488
      0.692326    0.73248096  0.4686703   0.641161  ]
    Est. explained variance: 0.69 (+/- 0.19)
    
    estimators: 80
    explained variance scores for k=10 fold validation: [ 0.59339318  0.69617323  0.7540893   0.66273855  0.78962986  0.82289323
      0.7146955   0.73390139  0.4159202   0.63237189]
    Est. explained variance: 0.68 (+/- 0.22)
    
    


```python
# and plot...
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(estimators,mean_rfrs,marker='o',
       linewidth=4,markersize=12)
ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,
                facecolor='green',alpha=0.3,interpolate=True)
ax.set_ylim([-.2,1])
ax.set_xlim([0,80])
plt.title('Expected Variance of Random Forest Regressor')
plt.ylabel('Expected Variance')
plt.xlabel('Trees in Forest')
plt.grid()
plt.show()
```


![png](output_9_0.png)



```python
import sklearn.feature_selection as fs # feature selection library in scikit-learn
train = pd.read_csv('C:/Users/1/Documents/House_Prices/train.csv') # get the training data again just in case
# first, let's include every feature that has data for all 1460 houses in the data set...
included_features = [col for col in list(train)
                    if len([i for i in train[col].T.notnull() if i == True])==1460
                    and col!='SalePrice' and col!='id']
# define the training data X...
X = train[included_features] # the feature data
Y = train[['SalePrice']] # the target
yt = [i for i in Y['SalePrice']] # the target list 
# and the data for the competition submission...
X_test = test[included_features]
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype=='object':
        X = getObjectFeature(X, col)
X.head()
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>...</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>0</td>
      <td>8450</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>0</td>
      <td>9600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>0</td>
      <td>11250</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>0</td>
      <td>9550</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>0</td>
      <td>14260</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>




```python
mir_result = fs.mutual_info_regression(X, yt) # mutual information regression feature ordering
feature_scores = []
for i in np.arange(len(included_features)):
    feature_scores.append([included_features[i],mir_result[i]])
sorted_scores = sorted(np.array(feature_scores), key=lambda s: s[1], reverse=True) 
print(np.array(sorted_scores))
```

    C:\Users\1\Anaconda3\lib\site-packages\sklearn\utils\validation.py:429: DataConversionWarning: Data with input dtype int64 was converted to float64 by the scale function.
      warnings.warn(msg, _DataConversionWarning)
    

    [['OverallQual' '0.506046629814']
     ['GrLivArea' '0.36760566186']
     ['GarageCars' '0.366811824525']
     ['KitchenQual' '0.340065826877']
     ['ExterQual' '0.335168494292']
     ['GarageArea' '0.285066720375']
     ['TotalBsmtSF' '0.263879802983']
     ['FullBath' '0.255862561265']
     ['YearBuilt' '0.253717102067']
     ['1stFlrSF' '0.2435626587']
     ['YearRemodAdd' '0.199304961569']
     ['Foundation' '0.194986477888']
     ['TotRmsAbvGrd' '0.182809280697']
     ['Fireplaces' '0.179634233741']
     ['HeatingQC' '0.165440708958']
     ['BsmtFinSF1' '0.133643728692']
     ['OverallCond' '0.114606612142']
     ['MSZoning' '0.114167221463']
     ['2ndFlrSF' '0.107459816174']
     ['SaleType' '0.0913867773997']
     ['LotArea' '0.0904150024111']
     ['LotShape' '0.0887408573726']
     ['HalfBath' '0.0765938759284']
     ['BedroomAbvGr' '0.0738739595044']
     ['SaleCondition' '0.0736321662293']
     ['HouseStyle' '0.060265920624']
     ['Neighborhood' '0.0589694797129']
     ['CentralAir' '0.0579373813843']
     ['OpenPorchSF' '0.0553469098164']
     ['MSSubClass' '0.0551796044418']
     ['BsmtUnfSF' '0.0497728223253']
     ['WoodDeckSF' '0.0449793742849']
     ['Exterior2nd' '0.042687541432']
     ['PavedDrive' '0.0426783472557']
     ['Exterior1st' '0.0315867698925']
     ['BldgType' '0.0312054268234']
     ['LandContour' '0.0242182823619']
     ['Heating' '0.0239990170796']
     ['ExterCond' '0.0205732396377']
     ['EnclosedPorch' '0.0201144075945']
     ['YrSold' '0.0194163717008']
     ['Condition1' '0.0170511944293']
     ['LandSlope' '0.0160852231867']
     ['LotConfig' '0.0159413159316']
     ['BsmtFullBath' '0.0155764582895']
     ['KitchenAbvGr' '0.0154571543702']
     ['RoofMatl' '0.0151202356308']
     ['RoofStyle' '0.0144330229723']
     ['Functional' '0.0116324702314']
     ['BsmtHalfBath' '0.0104424752804']
     ['LowQualFinSF' '0.00668183200136']
     ['BsmtFinSF2' '0.0057763748389']
     ['ScreenPorch' '0.00535657810098']
     ['Condition2' '0.00227692999264']
     ['Id' '0.0']
     ['Street' '0.0']
     ['Utilities' '0.0']
     ['3SsnPorch' '0.0']
     ['PoolArea' '0.0']
     ['MiscVal' '0.0']
     ['MoSold' '0.0']]
    


```python
# and plot...
fig = plt.figure(figsize=(13,6))
ax = fig.add_subplot(111)
ind = np.arange(len(included_features))
plt.bar(ind,[float(i) for i in np.array(sorted_scores)[:,1]])
ax.axes.set_xticks(ind)
plt.title('Feature Importances (Mutual Information Regression)')
plt.ylabel('Importance')
# plt.xlabel('Trees in Forest')
# plt.grid()
plt.show()
```


![png](output_12_0.png)



```python
# define a function to do the necessary model building....
def getModel(sorted_scores,train,numFeatures):
    included_features = np.array(sorted_scores)[:,0][:numFeatures] # ordered list of important features
    # define the training data X...
    X = train[included_features]
    Y = train[['SalePrice']]
    # transform categorical data if included in X...
    for col in list(X):
        if X[col].dtype=='object':
            X = getObjectFeature(X, col)
    # define the number of estimators to consider
    estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    mean_rfrs = []
    std_rfrs_upper = []
    std_rfrs_lower = []
    yt = [i for i in Y['SalePrice']]
    np.random.seed(11111)
    # for each number of estimators, fit the model and find the results for 8-fold cross validation
    for i in estimators:
        model = rfr(n_estimators=i,max_depth=None)
        scores_rfr = cross_val_score(model,X,yt,cv=10,scoring=explained_variance)
        mean_rfrs.append(scores_rfr.mean())
        std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
        std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting
    return mean_rfrs,std_rfrs_upper,std_rfrs_lower

# define a function to plot the model expected variance results...
def plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,numFeatures):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.plot(estimators,mean_rfrs,marker='o',
           linewidth=4,markersize=12)
    ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,
                    facecolor='green',alpha=0.3,interpolate=True)
    ax.set_ylim([-.2,1])
    ax.set_xlim([0,80])
    plt.title('Expected Variance of Random Forest Regressor: Top %d Features'%numFeatures)
    plt.ylabel('Expected Variance')
    plt.xlabel('Trees in Forest')
    plt.grid()
    plt.show()
    return
```


```python
# top 15...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,15)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,15)
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


![png](output_14_1.png)



```python
# top 20...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,20)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,20)
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


![png](output_15_1.png)



```python
# top 30...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,30)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,30)
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


![png](output_16_1.png)



```python
# top 40...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,40)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,40)
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


![png](output_17_1.png)



```python
# top 50...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,50)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,50)
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


![png](output_18_1.png)



```python
# build the model with the desired parameters...
numFeatures = 40 # the number of features to inlcude
trees = 60 # trees in the forest
included_features = np.array(sorted_scores)[:,0][:numFeatures]
# define the training data X...
X = train[included_features]
Y = train[['SalePrice']]
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype=='object':
        X = getObjectFeature(X, col)
yt = [i for i in Y['SalePrice']]
np.random.seed(11111)
model = rfr(n_estimators=trees,max_depth=None)
scores_rfr = cross_val_score(model,X,yt,cv=10,scoring=explained_variance)
print('explained variance scores for k=10 fold validation:',scores_rfr)
print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
# fit the model
model.fit(X,yt)
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    

    explained variance scores for k=10 fold validation: [ 0.85243869  0.88447151  0.92242765  0.76855337  0.8893922   0.88641444
      0.8908044   0.88791313  0.82918934  0.8608892 ]
    Est. explained variance: 0.87 (+/- 0.08)
    




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=60, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
# let's read the test data to be sure...
test = pd.read_csv('C:/Users/1/Documents/House_Prices/test.csv')
```


```python
# re-define a function to convert an object (categorical) feature into an int feature
# 0 = most common category, highest int = least common.
def getObjectFeature(df, col, datalength=1460):
    if df[col].dtype!='object': # if it's not categorical..
        print('feature',col,'is not an object feature.')
        return df
    else:
        df1 = df
        counts = df1[col].value_counts() # get the counts for each label for the feature
#         print(col,'labels, common to rare:',counts.index.tolist()) # get an ordered list of the labels
        df1[col] = [counts.index.tolist().index(i) 
                    if i in counts.index.tolist() 
                    else 0 
                    for i in df1[col] ] # do the conversion
        return df1 # make the new (integer) column from the conversion
```


```python
# apply the model to the test data and get the output...
X_test = test[included_features]
for col in list(X_test):
    if X_test[col].dtype=='object':
        X_test = getObjectFeature(X_test, col, datalength=1459)
# print(X_test.head(20))
y_output = model.predict(X_test.fillna(0)) # get the results and fill nan's with 0
print(y_output)
```

    C:\Users\1\Anaconda3\lib\site-packages\ipykernel\__main__.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    

    [ 128844.16666667  160644.16666667  181346.41666667 ...,  158252.5
      116729.58333333  234192.5       ]
    


```python
# define the data frame for the results
saleprice = pd.DataFrame(y_output, columns=['SalePrice'])
# print(saleprice.head())
# saleprice.tail()
results = pd.concat([test['Id'],saleprice['SalePrice']],axis=1)
results.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>128844.166667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>160644.166667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>181346.416667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>183005.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>205217.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# and write to output
results.to_csv('housepricing_submission.csv', index = False)
```


```python

```
