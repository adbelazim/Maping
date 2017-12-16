from keras.models import load_model
import keras.utils

import sklearn.metrics
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np

import os
import sys

import utils
import plot_utils


def train_svm(path_results, x_data, y_data, args):
	path_params = path_results + "/SVM/Metrics/params_test_" + str(args.step) + ".txt"

	params = eval(open(path_params).read())

	if "gamma" in params:
		classifier = SVC(kernel=params["kernel"], C=params["C"], gamma = params["gamma"],probability = True, max_iter = 10000, class_weight='balanced')

	else:
		classifier = SVC(kernel=params["kernel"], C=params["C"], probability = True, max_iter = 10000, class_weight='balanced')

	spliter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=19993)
	fold = 0
	for train_index, test_index in spliter.split(x_data, y_data):
		if str(fold) == str(args.fold):
			x_train = x_data[train_index]
			x_test = x_data[test_index]

			y_train = y_data[train_index] 
			y_test = y_data[test_index]

			classifier.fit(x_train, y_train)

		fold += 1

	predictions = classifier.predict(x_test)

	print(sklearn.metrics.classification_report(y_test, predictions))

	return classifier

def create_model(path_results, args):
	"""Carga la arquitectura del modelo y los pesos a esta.

	Necesita del path_results que se desee analizar

	"""

	path_model = path_results + "/Models/" + str(args.model) + "_" + str(args.step) + ".h5"
	path_weights = path_results + "/Models/" + str(args.model) + "_" + str(args.step) + "weights.hdf5"

	path_model_aux = path_results + "/Models/" + str(args.model) + "aux_" + str(args.step) + ".h5"
	path_weights_aux = path_results + "/Models/" + str(args.model) + "aux_" + str(args.step) + "weights.hdf5"

	print("path_model", path_model)
	print("path_weights", path_weights)

	model = load_model(path_model)
	model.load_weights(path_weights, by_name=False)

	model_aux = load_model(path_model_aux)
	model_aux.load_weights(path_weights_aux, by_name=False)

	print("Model loaded")

	return model, model_aux

def prepare_data(x_cbfv_data, x_cbfv_signal, args):
	"""Cambia el shape de la senial de entrada para que pueda ser predecida por el modelo.

	"""

	min_cbfv = np.amin(x_cbfv_data)
	max_cbfv = np.amax(x_cbfv_data)

	x_cbfv_normalize = utils.normalize(x_cbfv_data, min_cbfv, max_cbfv)
	x_cbfv_signal_normal = utils.normalize(x_cbfv_signal, min_cbfv, max_cbfv)

	if args.model == "resnet" or args.model == "fcn":
		x_cbfv_data = utils.prepare_data_cnn(x_cbfv_data, args)
		x_cbfv_ready = utils.prepare_data_cnn_signal(x_cbfv_signal_normal, args)

	else:
		x_cbfv_data = utils.prepare_data_lstm(x_cbfv_data, args)
		x_cbfv_ready = utils.prepare_data_lstm_signal(x_cbfv_signal_normal, args)

	return x_cbfv_data, x_cbfv_ready


def generate_invariant_data(path_data, x_cbfv_data, args):

	subjects = os.listdir(path_data)
	print("subjects",subjects)

	for subject in subjects:
		if subject == args.subject:
			#DY subject
			cbfv_signal = utils.read_cbfv_file(path_data + "/" + str(args.subject))
			#cambia frecuencia de muestreo a 0.6
			cbfv_signal = utils.lineal_interpolation(cbfv_signal)
			print("shape pre", cbfv_signal.shape)
			x_cbfv_data, x_cbfv_signal = prepare_data(x_cbfv_data, cbfv_signal, args)

	return x_cbfv_data, x_cbfv_signal


def generate_variant_data(path_data_nc, path_data_hc, x_cbfv_data, args):

	subjects = os.listdir(path_data_nc)
	print("subjects",subjects)

	for subject in subjects:
		if subject == args.subject:
			#DY subject
			cbfv_signal_nc = utils.read_cbfv_file(path_data_nc + "/" + str(args.subject))
			cbfv_signal_hc = utils.read_cbfv_file(path_data_hc + "/" + str(args.subject))

			print("len nc",len(cbfv_signal_nc))
			print("len hc",len(cbfv_signal_hc))

			if args.stick == "1":
				cbfv_signal = np.concatenate((cbfv_signal_nc[(len(cbfv_signal_nc)/2):len(cbfv_signal_nc)], cbfv_signal_hc[0:(len(cbfv_signal_hc)/2)]),axis = 0)
			else:
				cbfv_signal = np.concatenate((cbfv_signal_nc[0:(len(cbfv_signal_nc)/2)], cbfv_signal_hc[(len(cbfv_signal_hc)/2):len(cbfv_signal_hc)]),axis = 0)

			print("len pegada",len(cbfv_signal))

			#cambia frecuencia de muestreo a 0.6
			cbfv_signal = utils.lineal_interpolation(cbfv_signal)
			print("shape pre", cbfv_signal.shape)
			x_cbfv_signal = prepare_data(x_cbfv_data, cbfv_signal, args)

	return x_cbfv_signal

def select_model(path_results):
	path_metrics = path_results + "/Metrics/"

	kappas = []	

	for step in range(50, 550, 50):
		with open(path_metrics + "kappa_test_" + str(step) + ".txt") as f:
			kappa = f.readline()
		element = {"kappa": str(kappa),"step": str(step)}
		kappas.append(element)

	seq_kappa = [x["kappa"] for x in kappas]

	steps = []

	for x in kappas:
		if x["kappa"] == max(seq_kappa):
			steps.append(x["step"])

	return min(steps)


def maping(path, path_env):

	#data is in Results/args.mode/args.conjunt/5fold/args.model_numberclass_500epochs_args.train_data_foldi

	#LECTURA DE DATOS PARA PODER PLOTEAR MAPA
	#EXTRACCION DE CARACTERISTICAS GENERALES DE LOS DATOS. NORMALIZACION, BATCH_SEIZE, NUMB_CLASS
	args = utils.get_args()

	#get the path of data to train and validate the model
	train_data_path = path_env + "/" + args.train_dataset_path + "/" + args.mode
	val_data_path = path_env + "/" + args.val_dataset_path + "/" + args.mode

	#root path_save
	path_save = path_env + "/" + args.path_save + "/" + args.mode

	train_dataset = train_data_path + "/" + args.train_data
	val_dataset = val_data_path + "/" + args.val_data

	y_train_real, x_train_abp, x_train_cbfv = utils.read_multivariate_data(train_dataset + '/' + 'signals.txt')
	#read hipercapnia data
	y_val_real, x_val_abp, x_val_cbfv = utils.read_multivariate_data(val_dataset + '/' + 'signals.txt')

	if args.conjunt == "Mix":
		x_cbfv = np.concatenate((x_train_cbfv,x_val_cbfv),axis=0)
	else:
		x_cbfv = x_train_cbfv

	variant_mode = str(args.train_data).split("_")[4]
	print("variant_mode", variant_mode)

	#from data choose batch size
	batch_size = min(x_train_cbfv.shape[0]/10, 16)

	#get the number of classes in the dataset
	nb_classes_train = len(np.unique(y_train_real))
	nb_classes_test = len(np.unique(y_val_real))
	nb_classes = nb_classes_train

	#number of classes in train and test must to be the same.
	assert nb_classes_train == nb_classes_test

	##DESDE ACA INICIA EL TRABAJO DE MAPEO

	path_results = path_env + "/Results/" + str(args.mode) + "/" + str(args.conjunt) + "/5fold/" + str(args.model) + "_" + str(nb_classes) + "class" + "_500epochs_" + str(args.train_data) + "_fold" + str(args.fold)
	
	print("path_results", path_results)

	print("step pre",args.step)

	step_model = select_model(path_results)
	args.step = step_model

	print("step post",args.step)

	#load trained model
	model, model_aux = create_model(path_results, args)

	#path for data nc and hc
	path_data_nc = path + "/nc_data/"
	path_data_hc = path + "/hc_data/"

	#read invariant signal in nc to predict
	if args.mode == "Invariant":
		x_cbfv_data, signal_to_predict = generate_invariant_data(path_data_nc, x_cbfv, args)
	else:#variant and all
		x_cbfv_data, signal_to_predict = generate_variant_data(path_data_nc, path_data_hc, x_cbfv, args)

	print("signal_to_predict",signal_to_predict.shape)

	#sys.exit("test")

	#predict signal with normal model and model_aux
	model_prediction = model.predict(signal_to_predict, batch_size=batch_size)
	model_aux_prediction = model_aux.predict(signal_to_predict, batch_size=batch_size)
	#feature_space = model_aux.predict(x_cbfv_data, batch_size=batch_size)

	#print("model_aux_prediction",len(model_aux_prediction[0]))


	#CARGAR DATOS TSNE
	#PLOTEAR CON SENIAL NUEVA

	path_features = path_results + "/Weights/" + str(args.step) + ".csv"
	x_data, y_data = utils.read_weights(path_features)

	plot_utils.plot_visualization(path_results, x_data, y_data, variant_mode, nb_classes, model_aux_prediction,args)


	#THIS IS FOR BETTER PREDICTION

	#load svm model
	svm_model = train_svm(path_results, x_data, y_data, args)

	#predict signal with feature extraction of deep model
	prediction_svm = svm_model.predict(model_aux_prediction)
	print("prediction_svm", prediction_svm.shape)
	print("class of prediction", prediction_svm)
	print(prediction_svm[0])


	


if __name__ == "__main__":
	#example execution
	#python maping.py -m Variant -c Mix -train_data abp_cbfv_nc_4class_variantbi_freq0.6 -val_data abp_cbfv_hc_4class_variantbi_freq0.6 -model resnet -f 1 -step 500 -subject DY0000.txt
	#python maping.py -m Invariant -c Mix -train_data abp_cbfv_nc_2class_freq0.6 -val_data abp_cbfv_hc_2class_freq0.6 -model resnet -f 1 -step 500 -subject DY0000.txt


	path = os.getcwd()
	path_env, path_source = os.path.split(path)

	maping(path, path_env)


























