from pandas import read_csv
from numpy import array,arange
url300='https://raw.githubusercontent.com/ccakiroglu/CantileverSoldierPile/main/SoldierPile5varsKs300.csv'
url200='https://raw.githubusercontent.com/ccakiroglu/CantileverSoldierPile/main/SoldierPile5varsKs200.csv'
#sutunlar = [r'$b$', r'$h$', r'$t$', r'$l$', r'$f_y$', r'$f_c$', r'$N$']
sutunlar = [r'$\mathbf{L}$',r'$\mathbf{\gamma}$',r'$\mathbf{\phi}$',r'$\mathbf{q}$',r'$\mathbf{Cost}$', r'$\mathbf{D}$']
colnames = [r'$L$',r'$\gamma$',r'$\phi$',r'$q$',r'$Cost$', r'$D$']
vericercevesi = read_csv(url200,header=0, names=sutunlar)
dizimiz = vericercevesi.values
X = dizimiz[:,0:5]
y = dizimiz[:,5]

X=vericercevesi[[r'$\mathbf{L}$',r'$\mathbf{\gamma}$',r'$\mathbf{\phi}$',r'$\mathbf{q}$',r'$\mathbf{Cost}$']]
y=vericercevesi[[r'$\mathbf{D}$']]
import xgboost
import shap
from catboost import CatBoostRegressor
from matplotlib import pyplot
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=25)
pyplot.rcParams.update({'font.size': 45})
pyplot.rcParams['text.latex.preamble']=r"\usepackage{amsmath}\boldmath"
# train an XGBoost model
model = CatBoostRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.TreeExplainer(model)
#explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X)

#shap.initjs()
#shap.plots.force(shap_values[0], matplotlib=True, show=False)
#pyplot.xlabel('')
#pyplot.xlabel(r'$b$')
#pyplot.ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for$')
pyplot.tight_layout()
#shap.plots.beeswarm(shap_values, show=False,color_bar_label=r'$Feature\hspace{0.5em} value$' )
#shap.summary_plot(shap_values,X)
#shap.plots.beeswarm(shap_values)
#shap.plots.scatter(shap_values[:,r'$r$'],show=False, color=shap_values)#color_bar_labels
#shap.plots.scatter(shap_values[:,1],show=False, color=shap_values)#color_bar_labels
fig, ax = pyplot.gcf(), pyplot.gca()
ax.set_xlabel(r'$\mathbf{Cost}$', fontdict={"size":60})
#ax.set_xticklabels(Fontsize=25)
#ax.tick_params(axis='x', labelsize=45)
pyplot.tick_params(axis='both', which='major', labelsize=60)
#ax.set_xticklabels(fontdict={"size":20})
ax.set_ylabel(r'$\mathbf{SHAP \hspace{0.5em}value\hspace{0.5em} for\hspace{0.5em}Cost}$', fontdict={"size":60})
pyplot.tight_layout()
shap.dependence_plot(4, shap_values, X, ax=ax)
#shap.dependence_plot(r'$r$', shap_values, X, interaction_index=None)
#ax.set_yticks(array([-300,-200,-100,0,100,200, 300, 400,500]))
#ax.set_xticks(array([0,250,500,750,1000,1250,1500,1750,2000]))
#ax.tick_params(axis='y', labelsize=25)
#ax.tick_params(axis='x', labelsize=25)
#ax.set_xlabel(r'$h$',fontdict={"size":25})
#pyplot.savefig('G:\\My Drive\\Papers\\2022\\CylindricalWall\\IMAGES\\CatBoostSHAPfeatDepR.svg')
#print(shap_values.shape)
