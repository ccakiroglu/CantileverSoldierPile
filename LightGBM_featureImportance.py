import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgbm
from pandas import read_csv, DataFrame
from numpy import absolute,arange,mean,std,argsort,sqrt, sum
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor 
#from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from matplotlib import pyplot 
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=25,weight='bold')
pyplot.rcParams['text.latex.preamble']=r"\usepackage{amsmath}\boldmath"
colnames = [r'$\mathbf{L}$',r'$\mathbf{\gamma}$',r'$\mathbf{\phi}$',r'$\mathbf{q}$',r'$\mathbf{Cost}$', r'$\mathbf{D}$']
url300='https://raw.githubusercontent.com/ccakiroglu/CantileverSoldierPile/main/SoldierPile5varsKs300.csv'
url200='https://raw.githubusercontent.com/ccakiroglu/CantileverSoldierPile/main/SoldierPile5varsKs200.csv'
df = read_csv(url300,header=0,names=colnames)
data=df.values
print('data.shape:',data.shape)
# split into inputs and outputs
#X, y = df.iloc[:, :-1], df.iloc[:, -1]
X, y = data[:, :-1], data[:, -1]
print('X.shape:', X.shape,'y.shape', y.shape)
# split into train test sets
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

LGBMModel = LGBMRegressor()
X_standardized = scaler.transform(X)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++
cv = KFold(n_splits=10, shuffle=True, random_state=1)
#Here we list the parameters we want to tune
space = dict()
#space['n_estimators'] = arange(1,600,50)
space['n_estimators'] = [1000]
space['max_depth']=[2]
#space['min_child_weight']=[1, 5, 10]
#space['learning_rate']=arange(0.1, 4, 0.5)
#space['learning_rate']=[.03, 0.05, .07]
#space['learning_rate']=[0.02, 0.05, 0.1, 0.2]
space['learning_rate']=[0.2]
#i=1000
#r2scoresLR02=[]
#while i<2025:
#    space['n_estimators'] = [i]
    #space['loss']=['linear', 'square', 'exponential']
search = GridSearchCV(LGBMModel, space, n_jobs=-1, cv=cv, refit=True)
result = search.fit(X_train, y_train)
best_model = result.best_estimator_
yhat_test = best_model.predict(X_test)
yhat_train = best_model.predict(X_train)
#    r2scoresLR02.append(r2_score(y_test, yhat_test))
#    i+=25
#pyplot.scatter(yhat_train,y_train, marker='s',facecolor='seagreen',edgecolor='seagreen', label=r'$Training\hspace{0.5em}set$')#original
#pyplot.scatter(yhat_test,y_test, marker='s',facecolor='none',edgecolor='blue', label=r'$Test\hspace{0.5em}set$')#original
#x=arange(25,625,25)
#x=arange(1000,2025,25)
#pyplot.plot(x,r2scoresLR02, color='blue', label=r'$L_r=0.02$')
#pyplot.plot(x,r2scoresLR05, color='red', dashes=[5,2,2,2], label=r'$L_r=0.05$')
#pyplot.plot(x,r2scoresLR1, color='green', dashes=[5,2], label=r'$L_r=0.1$')
#pyplot.plot(x,r2scoresLR2, color='yellow', dashes=[2,2], label=r'$L_r=0.2$')
#pyplot.xticks(arange(1000,2100,100))
#pyplot.title(r'$max\_depth=5$')
#xk=[0,12500];yk=[0,12500];
#pyplot.plot(xk,yk, color='black')
#pyplot.grid(True)
#pyplot.xlabel(r'$Number\hspace{0.5em}of\hspace{0.5em}trees$')
#pyplot.ylabel(r'$R^2\hspace{0.5em}score$')
#pyplot.legend()
#pyplot.tight_layout()
#pyplot.savefig('G:\\My Drive\\Papers\\Papers\\CFST_ML\\IMAGES\\Overfitting.svg')
#pyplot.show()
#R2=r2_score(y_test, yhat)#original
print('MSE train= ',mean_squared_error(y_train, yhat_train))
print('RMSE train= ',sqrt(mean_squared_error(y_train, yhat_train)))
print('MAE train= ',mean_absolute_error(y_train, yhat_train))
print('R2 train:',r2_score(y_train, yhat_train))
print('MSE test= ',mean_squared_error(y_test, yhat_test))
print('RMSE test= ',sqrt(mean_squared_error(y_test, yhat_test)))
print('MAE test= ',mean_absolute_error(y_test, yhat_test))
print('R2 test:',r2_score(y_test, yhat_test))#original
print('Best parameters are',search.best_params_)#original
def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names[0:5])
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    #plt.figure(figsize=(10,8))
    plt.figure()
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    #plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel(r'$\mathbf{Feature\hspace{0.5em}Importance[\%]}$',fontsize=20)
    plt.ylabel(r'$\mathbf{Feature\hspace{0.5em}Names}$', fontsize=20)
    plt.xticks(arange(0,50,10),fontsize=20)
    plt.yticks(fontsize=20)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()
sum_of_importances=sum(best_model.feature_importances_)
print(100*best_model.feature_importances_/sum_of_importances)
#print("The sum of feature importances = ",sum(best_model.feature_importances_))
plot_feature_importance(100*best_model.feature_importances_/sum_of_importances,colnames,'LightGBM')
