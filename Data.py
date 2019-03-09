from sklearn.datasets import load_boston
#from sklearn.datasets import laod_boston

# [func for fuc in dir(sklearn.datasets) in fuc.startswitch("load")]

boston_data = load_boston()

dir(boston_data)

print(boston_data['DESCR'])
