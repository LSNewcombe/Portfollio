import pickle as pic
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_curve, auc, roc_auc_score
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPool1D, AveragePooling1D
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import TensorBoard
from itertools import cycle

class NeuralNetwork():

	def __init__(self):
		 # ------DEFINE THE INPUT DATA SET TO USE WITH THE NETWORK-----------------------------------------------
		fileloc = r'PreparedFeatures\AllFeaturesNOPEAKS.xlsx'
		picWristOutput = r'PreparedFeatures\PickleWrist.pickle'
		picWristLabel = r'PreparedFeatures\PickleWristLabel.pickle'
		picWaistOutput = r'PreparedFeatures\PickleWaist.pickle'
		picWaistLabel = r'PreparedFeatures\PickleWaistLabel.pickle'
		np.set_printoptions()
		np.set_printoptions(formatter={"float_kind": "{:f}".format})

		#self.dataWaist = pd.read_excel(fileloc, sheet_name ='Waist', na_values=['NA'], usecols = 'A:CH')
		#self.waistLabel = pd.read_excel(fileloc, sheet_name ='Waist', na_values=['NA'], usecols = 'CI')

		# Import data from the Excel Spreadsheet of Feature Data
		try:
			self.dataWrist = pic.load(open(picWristOutput, 'rb'))
		except (EOFError, OSError, IOError, FileNotFoundError):
			self.dataWrist = pd.read_excel(fileloc, sheet_name='Wrist', na_values=['NA'], usecols='A:BP')
			pic.dump(self.dataWrist, open(picWristOutput, 'wb'))
		
		try:
			self.wristLabel = pic.load(open(picWristLabel, 'rb'))
		except (EOFError, OSError, IOError, FileNotFoundError):
			self.wristLabel = pd.read_excel(fileloc, sheet_name='Wrist', na_values=['NA'], usecols='BQ')
			pic.dump(self.wristLabel, open(picWristLabel, 'wb'))

		try:
			self.dataWaist = pic.load(open(picWaistOutput, 'rb'))
		except (EOFError, OSError, IOError, FileNotFoundError):
			self.dataWaist = pd.read_excel(fileloc, sheet_name='Waist', na_values=['NA'], usecols='A:BP')
			pic.dump(self.dataWaist, open(picWaistOutput, 'wb'))
		
		try:
			self.waistLabel = pic.load(open(picWaistLabel, 'rb'))
		except (EOFError, OSError, IOError, FileNotFoundError):
			self.waistLabel = pd.read_excel(fileloc, sheet_name='Waist', na_values=['NA'], usecols='BQ')
			pic.dump(self.waistLabel, open(picWaistLabel, 'wb'))

		self.xTrainInput = None
		self.xTestInput = None
		self.yTrainInput = None
		self.yTestInput = None
		self.classes = ["Walking","Walking Upstairs","Walking Downstairs","Walking Circle","Sitting","Standing","Lying","Pickup Object","Place Down Object","Drinking Water",
		"Opening Door","Closing Door","Stand to Sit","Sit to Lie","Lie to Sit","Sit to Stand","Bending Over to Standing","Standing to Bending Over"]

	def partitionData(self):
		# partition data sets into training and testing inputs and outputs - fixed size partition split
		self.x_trainWrist, self.x_testWrist, self.y_trainWrist, self.y_testWrist, self.x_trainWaist, self.x_testWaist, self.y_trainWaist, self.y_testWaist = train_test_split(self.dataWrist, self.wristLabel, self.dataWaist, self.waistLabel, shuffle=True, test_size=0.1)

		# For each value convert the data using the to_numpy function
		self.x_trainWrist = self.x_trainWrist.to_numpy()
		self.y_trainWrist = self.y_trainWrist.to_numpy()
		self.x_testWrist = self.x_testWrist.to_numpy()
		self.y_testWrist = self.y_testWrist.to_numpy()

		self.x_trainWaist = self.x_trainWaist.to_numpy()
		self.y_trainWaist = self.y_trainWaist.to_numpy()
		self.x_testWaist = self.x_testWaist.to_numpy()
		self.y_testWaist = self.y_testWaist.to_numpy()

		data = pd.DataFrame(self.y_trainWrist)
		data.to_excel(r'LogsTF\ytrain.xlsx')

	def setInputData(self, sensor):
		self.sensor = sensor

		if sensor == 'Wrist':
			self.xTrainInput = self.x_trainWrist
			self.yTrainInput = self.y_trainWrist
			self.xTestInput = self.x_testWrist
			self.yTestInput = self.y_testWrist
		elif sensor == 'Waist':
			self.xTrainInput = self.x_trainWaist
			self.yTrainInput = self.y_trainWaist
			self.xTestInput = self.x_testWaist
			self.yTestInput = self.y_testWaist
		
		smote = SMOTE()

		self.xTrainInput, self.yTrainInput = smote.fit_sample(self.xTrainInput, self.yTrainInput)

	def trainNetwork(self, activationFunction):
		model = Sequential()
		self.epochs = 200
		verbose, batch_size = 2, 32
		self.xTrainInput = np.expand_dims(self.xTrainInput, axis = 2)
		self.xTestInput = np.expand_dims(self.xTestInput, axis = 2)
		n_timesteps, n_features = self.xTrainInput.shape[1], self.xTrainInput.shape[2]
		
		model = Sequential()
		model.add(Convolution1D(filters=64, kernel_size=3, activation= activationFunction, use_bias = True, input_shape=(n_timesteps,n_features)))
		model.add(MaxPool1D(pool_size=2))
		model.add(Convolution1D(filters=64, kernel_size=3, activation= activationFunction))
		model.add(Dropout(0.5))
		model.add(MaxPool1D(pool_size=2))
		model.add(Flatten())
		model.add(Dense(50, activation= activationFunction))
		model.add(Dense(18, activation='softmax'))
		model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mean_squared_error', 'mae', 'mape', 'cosine'])

		tensorboard = TensorBoard(
  			log_dir=r'Results\logs'+str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) + 'BatchSize33' ,
 			histogram_freq=1,
  			write_graph= True,
		)

		keras_callbacks = [tensorboard]
		self.history = model.fit(self.xTrainInput, self.yTrainInput, epochs=self.epochs, batch_size=batch_size, verbose=verbose, validation_split = 0.1, callbacks=keras_callbacks)
		
		# evaluate model
		self.yPred = model.predict_classes(self.xTestInput, batch_size=batch_size, verbose=0)
		self.yScore = model.predict_proba(self.xTestInput, batch_size=batch_size, verbose=0)
		self.fit =  model.predict(self.xTestInput, batch_size=batch_size, verbose=0)
		plot_model(model, show_shapes = True, expand_nested = True, to_file=r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\Model ' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) + '.png')
		

		#Calculate Metrics
		self.calculateMetrics(activationFunction)

		#Caluclate ROC
		self.calculateROC(activationFunction)

		#Plot Accuracy of the Model
		self.plotAccuracy(activationFunction)

		#Plot Loss of the Model
		self.plotLoss(activationFunction)

		#plot MSE
		self.plotMSE(activationFunction)

		#plot MAE
		self.plotMAE(activationFunction)

		#plot MAPE
		self.plotMAPE(activationFunction)

		#plot cosine proximity
		self.plotCosine(activationFunction)

		loss, accuracy, mse, mae, mape, cosine= model.evaluate(self.xTestInput, self.yTestInput, batch_size=batch_size, verbose=0)
		return loss, accuracy
	
	def calculateMetrics(self, activationFunction):
		classReport = classification_report(self.yTestInput, self.yPred, target_names= self.classes, output_dict= True, )

		df = pd.DataFrame(classReport).transpose()
		df.to_excel(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\Results '+ str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) +'.xlsx')
		conMat = tf.math.confusion_matrix(labels = self.yTestInput, predictions = self.yPred).numpy()
		con_mat_norm = np.around(conMat.astype('float') / conMat.sum(axis=1)[:, np.newaxis], decimals=2)

		con_mat_df = pd.DataFrame(con_mat_norm,
					 index = self.classes, 
					 columns = self.classes)
		plt.figure(figsize = (12,12))
		sb.heatmap(con_mat_df, annot=True,cmap= "Blues")
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\ConfMat' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction)+ '.png', bbox_inches = 'tight')
		plt.close()
		self.FP = con_mat_norm.sum(axis = 0) - np.diag(con_mat_norm)
		self.FN = con_mat_norm.sum(axis = 1) - np.diag(con_mat_norm)
		self.TP = np.diag(con_mat_norm)
		self.TN = con_mat_norm.sum() - (self.FP + self.FN + self.TP)

		# Sensitivity, hit rate, recall, or true positive rate
		self.TPR = self.TP/(self.TP+self.FN)
		# Specificity or true negative rate
		self.TNR = self.TN/(self.TN+self.FP) 
		# Precision or positive predictive value
		self.PPV = self.TP/(self.TP+self.FP)
		# Negative predictive value
		self.NPV = self.TN/(self.TN+self.FN)
		# Fall out or false positive rate
		self.FPR = self.FP/(self.FP+self.TN)
		# False negative rate
		self.FNR = self.FN/(self.TP+self.FN)
		# False discovery rate
		self.FDR = self.FP/(self.TP+self.FP)
		# Overall accuracy
		self.ACC = (self.TP+self.TN)/(self.TP+self.FP+self.FN+self.TN)
		# F1 Score
		self.F1 = 2 * (self.PPV * self.TPR / (self.PPV + self.TPR))
		#MCC
		self.MCC = (self.TP * self.TN) - (self.FP * self.FN) / np.sqrt((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))

		df = pd.DataFrame({'Sensitivity': self.TPR,
						   'Specificity': self.TNR,
						   'Precision':self. PPV,
						   'Negative Predicitve Value': self.NPV,
						   'False Positive Rate': self.FPR,
						   'False Negative Rate': self.FNR,
						   'False Discovery Rate': self.FDR,
						   'Accuracy': self.ACC,
						   'F1 Score': self.F1,
						   'MCC': self.MCC},
						   index = self.classes)

		df.to_excel(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\Results2 '+ str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) +'.xlsx')
	
	def calculateROC(self, activationFunction):
		yBinary = preprocessing.label_binarize(self.yTestInput, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
		n_classes = yBinary.shape[1]

		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(n_classes):
			fpr[i], tpr[i], _ = roc_curve(yBinary[:, i], self.yScore[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])

		# Compute micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = roc_curve(yBinary.ravel(), self.yScore.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(n_classes):
   			mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

		# Finally average it and compute AUC
		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

		# Plot all ROC curves
		plt.figure(figsize = (12,12))
		plt.plot(fpr["micro"], tpr["micro"],
        		 label='Micro-Average ROC curve (AUC= {0:0.2f})'
              		 ''.format(roc_auc["micro"]),
         		color='deeppink', linestyle=':', linewidth=4)

		plt.plot(fpr["macro"], tpr["macro"],
         		label='Macro-Average ROC curve (AUC = {0:0.2f})'
               		''.format(roc_auc["macro"]),
         		color='navy', linestyle=':', linewidth=4)

		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue','green', 'darkred', 'dodgerblue', 'violet', 'purple', 'darkcyan', 'lawngreen', 'indigo', 'lightseagreen', 'goldenrod', 'indigo', 'crimson', 'darkgrey'])
		for i, color in zip(range(n_classes), colors):
   			plt.plot(fpr[i], tpr[i], color=color, lw=2,
           		  label= str(self.classes[i]) + ' (AUC = {1:0.2f})'.format(i, roc_auc[i]))

		plt.plot([0, 1], [0, 1], 'k--', lw=2)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve for Activities: ' + str(self.sensor) + ' Epochs: ' + str(self.epochs) + ' ' + str(activationFunction) + ' Batch Size: 32')
		plt.legend(loc="lower right")
		plt.savefig(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\ModelROC' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction)+ '.png', bbox_inches = 'tight')
		plt.close()


	
	def plotAccuracy(self, activationFunction):
		# Plot training & validation accuracy values
		plt.figure(figsize = (12,12))
		plt.plot(self.history.history['accuracy'])
		plt.plot(self.history.history['val_accuracy'])
		plt.title('Model Accuracy: ' + str(self.sensor) + ' Epochs: ' + str(self.epochs) + ' ' + str(activationFunction) + ' Batch Size: 32')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\ModelAcc' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) + '.png', bbox_inches = 'tight')
		plt.close()
		
	
	def plotLoss(self, activationFunction):
		# Plot training & validation loss values
		plt.figure(figsize = (12,12))
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('Model Loss: ' + str(self.sensor) + ' Epochs: ' + str(self.epochs) + ' ' + str(activationFunction) + ' Batch Size: 32')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\ModelLoss' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) + '.png', bbox_inches = 'tight')
		plt.close()
	
	def plotMSE(self, activationFunction):
		# Plot training & validation Mean Squared Errors 
		plt.figure(figsize = (12,12))
		plt.plot(self.history.history['mean_squared_error'])
		plt.plot(self.history.history['val_mean_squared_error'])
		plt.title('Model Mean Squared Error ' + str(self.sensor) + ' Epochs: ' + str(self.epochs) + ' ' + str(activationFunction) + ' Batch Size: 32')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\ModelMSE' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) + '.png', bbox_inches = 'tight')
		plt.close()

	def plotMAE(self, activationFunction):
		# Plot training & validation Mean Absolute Errors 
		plt.figure(figsize = (12,12))
		plt.plot(self.history.history['mae'])
		plt.plot(self.history.history['val_mae'])
		plt.title('Model Mean Absolute Error ' + str(self.sensor) + ' Epochs: ' + str(self.epochs) + ' ' + str(activationFunction) + ' Batch Size: 32')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\ModelMAE' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) + '.png', bbox_inches = 'tight')
		plt.close()

	def plotMAPE(self, activationFunction):
		# Plot training & validation Mean Absolute Percentage Errors 
		plt.figure(figsize = (12,12))
		plt.plot(self.history.history['mape'])
		plt.plot(self.history.history['val_mape'])
		plt.title('Model Mean Absolute Percentage Error ' + str(self.sensor) + ' Epochs: ' + str(self.epochs) + ' ' + str(activationFunction) + ' Batch Size: 32')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\ModelMAPE' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) + '.png', bbox_inches = 'tight')
		plt.close()

	def plotCosine(self, activationFunction):
		# Plot training & validation Cosine Proximity
		plt.figure(figsize = (12,12))
		plt.plot(self.history.history['cosine'])
		plt.plot(self.history.history['val_cosine'])
		plt.title('Model Cosine Proximity ' + str(self.sensor) + ' Epochs: ' + str(self.epochs) + ' ' + str(activationFunction) + ' Batch Size: 32')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig(r'Results\Epochs ' + str(self.epochs) + str(activationFunction) +'BatchSize32\ModelCosine' + str(self.sensor) + 'Epochs' + str(self.epochs) + str(activationFunction) + '.png', bbox_inches = 'tight')
		plt.close()
	
if __name__ == "__main__":
	neural_network = NeuralNetwork()
	neural_network.partitionData()
	neural_network.setInputData('Wrist')
	wristloss, wristaccuracy = neural_network.trainNetwork('relu')

	neural_network.setInputData('Waist')
	waistloss, waistaccuracy = neural_network.trainNetwork('relu')
	print(wristloss, wristaccuracy, waistloss, waistaccuracy)

	epochindex = ['Epoch', str(neural_network.epochs)]
	data = pd.DataFrame({'Wrist Loss': wristloss, 'Wrist Accuracy': wristaccuracy, 'Waist Loss': waistloss, 'Waist Accuracy': waistaccuracy}, index = epochindex)
	data.to_excel(r'Results\Epochs ' + str(neural_network.epochs) +'reluBatchSize32\Accuracy and Loss.xlsx')
	