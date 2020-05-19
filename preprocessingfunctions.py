
from sklearn.impute import *
import numpy as np
import pandas as pd

##verilen sutunlarda verilen değeri arar ve buldugunda sutunun ortalamasını yazar
def imput_mean_cols (dataset, cols,value):
	for col in cols:
		imputer=SimpleImputer(missing_values=value,strategy='mean')
		out=dataset[[col]]
		imputer=imputer.fit(out)
		dataset[[col]]=imputer.transform(out)
	return dataset

#verilen sutunlarda  verilen değeri arar buldugunda medyanını yazar
def imput_median_cols (dataset, cols,value):
	for col in cols:
		imputer=SimpleImputer(missing_values=value,strategy='median')
		out=dataset[[col]]
		imputer=imputer.fit(out)
		dataset[[col]]=imputer.transform(out)
	return dataset

#verilen sutunlarda verilen değeri arar buldugu satırı siler
def delete_cols_with_value(dataset,cols,value):
	for col in cols:
		dataset[[col]]=dataset[[col]].replace(value,np.NaN)
	dataset=dataset.dropna()
	return dataset
