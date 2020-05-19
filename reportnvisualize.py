#ClassificationReport
from sklearn.metrics import classification_report

y_pred=model.predict_classes(X_test)#model.predict(X_test)

print(classification_report(y_test, y_pred))



#Figure
fit=model.fit()
import matplotlib.pyplot as plt
# Plot the Loss Curves

plt.subplot(211)
plt.plot(fit.history['loss'], 'r')
plt.plot(fit.history['val_loss'], 'b')
plt.legend(['Training loss', 'Validation Loss'])
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.title('Loss Curves')
# Plot the Accuracy Curves
plt.subplot(212)
plt.plot(fit.history['accuracy'], 'r')
plt.plot(fit.history['val_accuracy'], 'b')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epochs ')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')

plt.savefig('modelfig')

#Confusion Matrix
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test, y_test)  # doctest: +SKIP
plt.show()  
