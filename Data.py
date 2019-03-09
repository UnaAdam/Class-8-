from sklearn.datasets import load_boston
#from sklearn.datasets import laod_boston

# [func for fuc in dir(sklearn.datasets) in fuc.startswitch("load")]

boston_data = load_boston()

dir(boston_data)
print(boston_data['DESCR'])
print(boston_data.keys())

print(boston_data['target'])
print(boston_data['feature_names'])

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston.head()

for col in boston:
   idx=boston.columns.get_loc(col)
   sns.set(rc={'figure.figsize':(11.7,8.27)})
   sns.distplot(boston[col],rug=False,bins=30).set_title("Histogram of {0}".format(col))
   plt.savefig("./{0}_{1}.png".format(idx,col), dpi=100)
   plt.close()


correlation_matrix = boston.corr().roud(2)
heatmap = sns.heatmap(data=correlation_matrix, annot=True)
myfig = heatmap.get_figure()
myfig.savefig('output.png')
plt.close()

