import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



column_names = ["sex", "length", "diameter", "height", "whole weight",
                "shucked weight", "viscera weight", "shell weight", "rings"]
data = pd.read_csv('abalone.csv', names=column_names,skiprows=1)

length=data['length']
diameter=data['diameter']
height=data['height']
whole_weight=data['whole weight']
shucked_weight=data['shucked weight']
viscera_weight=data['viscera weight']
shell_weight=data['shell weight']
rings=data['rings']
sex=data['sex']

print('----------Length----------')
print('Length Max')
print(length.max())
print('Length Min')
print(length.min())
print('Length Ortalama')
print(length.mean())
print('Length Max Standart Sapma')
print(length.std())


print('----------Diameter----------')
print('Diameter Max')
print(diameter.max())
print('Diameter Min')
print(diameter.min())
print('Diameter Ortalama')
print(diameter.mean())
print('Diameter Standart Sapma')
print(diameter.std())


print('----------Height----------')
print('Height Max')
print(height.max())
print('Height Min')
print(height.min())
print('Height Ortalama')
print(height.mean())
print('Height Standart Sapma')
print(height.std())


print('----------Whole Weight----------')
print('Whole Weight Max')
print(whole_weight.max())
print('Whole Weight Min')
print(whole_weight.min())
print('Whole Weight Ortalama')
print(whole_weight.mean())
print('Whole Weight Standart Sapma')
print(whole_weight.std())


print('----------Shucked Weight----------')
print('Shucked Weight Max')
print(shucked_weight.max())
print('Shucked Weight Min')
print(shucked_weight.min())
print('Shucked Weight Ortalama')
print(shucked_weight.mean())
print('Shucked Weight Standart Sapma')
print(shucked_weight.std())


print('----------Viscera Weight----------')
print('Viscera Weight Max')
print(viscera_weight.max())
print('Viscera Weight Min')
print(viscera_weight.min())
print('Viscera Weight Ortalama')
print(viscera_weight.mean())
print('Viscera Weight Standart Sapma')
print(viscera_weight.std())


print('----------Shell Weight----------')
print('Shell Weight Max')
print(shell_weight.max())
print('Shell Weight Min')
print(shell_weight.min())
print('Shell Weight Ortalama')
print(shell_weight.mean())
print('Shell Weight Standart Sapma')
print(shell_weight.std())


print('----------Rings----------')
print('Rings Max')
print(rings.max())
print('Rings Min')
print(rings.min())
print('Rings Ortalama')
print(rings.mean())
print('Rings Standart Sapma')
print(rings.std())

plt.scatter(length,diameter, c='blue', s=5)
plt.xlabel('Length')
plt.ylabel('Diameter')
plt.savefig('Length-Diameter.png')
plt.show()
plt.close()

plt.scatter(height,whole_weight, c='red', s=5)
plt.xlabel('Height')
plt.ylabel('Whole Weight')
plt.savefig('Height-Whole Weight.png')
plt.show()
plt.close()

plt.scatter(shucked_weight,viscera_weight, c='green', s=5)
plt.xlabel('Shucked Weight')
plt.ylabel('Viscera Weight')
plt.savefig('Shucked Weight-Viscera Weight.png')
plt.show()
plt.close()

plt.scatter(shell_weight,rings, c='gray', s=5)
plt.xlabel('Shell Weight')
plt.ylabel('Rings')
plt.savefig('Shell Weight-Rings.png')
plt.show()
plt.close()

_dataMatrix=data.to_numpy()
_data=[]

_data.extend(_dataMatrix[0:3340, 1:])

_test= []
_test.extend(_dataMatrix[3341:, 1:])

_class_pred = []
_class_pred.extend(_dataMatrix[3341:, 0])

_class_data = []
_class_data.extend(_dataMatrix[0:3340:, 0])

clf=tree.DecisionTreeClassifier()

clf=clf.fit(_data,_class_data)

_pred=clf.predict(_test)

print('----------Confusion Matrix----------')
print(confusion_matrix(_class_pred,_pred))
print('------------------------------------')
print('--------Classification Report-------')
print(classification_report(_class_pred,_pred,_dataMatrix[:,0]))
print('------------------------------------')

tree.plot_tree(clf)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("abaloneTree")

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)

plt.show(graph)
plt.savefig('Karar Agaci.png')
