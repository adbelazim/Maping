import numpy as np
import argparse


def get_args():
	"""Args given by command line.

	Return:
	Args given by command line. Some parameters has default behavior.
	
	"""

	parser = argparse.ArgumentParser()

	parser.add_argument("-m","--mode", type = str, default = "Variant", \
    						help = 'Data to process. Could be "All","Invariant" or "Variant". The first letter must to be upper')

	parser.add_argument("-c","--conjunt", type = str, default = "Mix", \
    						help = 'Mix for all training o Validation.')

	parser.add_argument("-train_data","--train_data", type = str, default = "abp_cbfv_nc_4class_variantbi_freq0.6", \
    						help = 'Folder of train dataset')
	
	parser.add_argument("-val_data","--val_data", type = str, default = "abp_cbfv_hc_4class_variantbi_freq0.6", \
    						help = 'Folder of val dataset')

	parser.add_argument("-train_dataset_path","--train_dataset_path", type = str, default = "Datasets_nc", \
    						help = 'Train data folder to train the network')
	
	parser.add_argument("-val_dataset_path","--val_dataset_path", type = str, default = "Datasets_hc", \
    						help = 'Validation data folder to prove the trained model')

	parser.add_argument("-path_save","--path_save", type = str, default = "Maping/Visualization/",\
    						help = "Path to save weights, model, loss, etc.")

	parser.add_argument("-model","--model", type = str, default = "resnet", \
    						help = "Name of the model to train. Could be: knn-dtw, resnet, lstm and fcn.")

	parser.add_argument("-f","--fold", type = str, default = "1", \
    						help = "Number of fold to analize")

	parser.add_argument("-step","--step", type = str, default = "500", \
    						help = "Step of selected model.")

	parser.add_argument("-subject","--subject", type = str, default = "HC0901.txt", \
    						help = "Subject to test the model.")

	parser.add_argument("-stick","--stick", type = str, default = "1", \
    						help = "how to stick nc and hc signals. Coulb be 1 or 2")


	args = parser.parse_args()

	return args

def lineal_interpolation(vector,freq=float(0.6)):
	"""Resamping a signal by a frequency.

	Parameters:
	vector -- a signal.
	freq -- frequency to resampling the signal.

	Return:
	A resampling signal.

	"""

	step = int(freq*5)
	vector = np.asarray(vector,dtype=np.float64)
	new_vector = np.zeros((len(vector)/step,1))
	aux = 0
	for index in range(0,len(vector)-1,step):
		if index/step != len(new_vector): #condicion de borde
			new_vector[index/step,0] = np.sum(vector[index:index+step])/step
		aux = index + step
	return new_vector

def read_weights(filename):
	"""Lectura de archivo que contiene matriz de pesos de ultima capa del modelo.

	Se utiliza numpy para lectura de archivo con delimitador ",".
	En este caso particular la matriz contiene 128 neuronas.

	Parametros:
	filename -- archivo .csv delimitado por comas

	"""

	data = np.loadtxt(filename, delimiter = ',')
	columns = len(data[0])
	label = data[:,columns-1]
	X = data[:,0:columns-1]

	# print(X.shape)
	# print(X[0])
	# print(label.shape)
	# print(label[0])

	return X, label.astype(int)

def read_cbfv_file(file_data):
	"""Read a file extracting only cbfv signal.

	A file contains the following data in each column:
		time_freq (0)
		abp (1)
		cbfv (2) #this case
 		etco2 (3)

	Parameters:
	file_data -- filename with data to read.
	
	Return:
	A list with cbfv signal.

	"""

	cbfv_aux = []
	with open(file_data,'r') as f:
		lines = f.readlines()
		for line in lines:
			columns = line.split('\t')
			cbfv_aux.append((columns[2]))

	return np.asarray(cbfv_aux,dtype=np.float64)

def read_multivariate_data(filename):
	"""Read label and two signal from filename.

	The format of the file is:
		label, signal1;signal2
		label, signal1;signal2
		...

	Parameters:
	filename -- filename of the data.

	Return:
	Labels, signal1(abp) and signal2(cbfv) in numpy array format.

	"""

	cbfv_data = []
	abp_data = []
	labels = []

	with open(filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			cbfv_vector = []
			abp_vector = []

			split_data = line.split(";")
			cbfv_line = split_data[1].split(",")

			abp_line = split_data[0].split(",")
			labels.append(abp_line[0])
			abp_line = abp_line[1:len(abp_line)]
			aux = 0

			for i in range(0,len(cbfv_line)):
				cbfv_vector.append(float(cbfv_line[i]))
				abp_vector.append(float(abp_line[i]))
			cbfv_data.append(cbfv_vector)
			abp_data.append(abp_vector)
	
	return np.asarray(labels, dtype=np.int32), np.asarray(abp_data,dtype=np.float64), np.asarray(cbfv_data,dtype=np.float64)


def normalize(x_data, min_data, max_data):
	"""Normalize data by mean and std.

	Parameters:
	x_data -- data to normalize.
	mean -- mean to normalize data
	std -- standar desviation to normalize data

	Return:
	Normalize x_data

	"""

	x_data = (x_data - min_data) / (max_data - min_data)

	#x_data = (x_data - mean)/(std)

	return x_data

def prepare_data_cnn_signal(x_data, args):
	"""Prepare data for 2D-CNN architecture.

	This functions return a 4D array for train a CNN architecture.
	Channels is the last element in the 4D array.

	Parameters:
	x_train -- data to reshape.

	Return:
	4D array of (samples, rows, cols, channels) this is the default of Keras implementation.

	"""

	if x_data.ndim == 2:
		x_data = x_data.reshape(x_data.shape[1],x_data.shape[0], 1, 1)
		
	#TO DO
	elif x_data.ndim == 3:
		x_data = x_data.reshape(x_data.shape[1],x_data.shape[0], 1)

	else: #4 dim
		x_data = x_data

	# if args.timesteps:
	# 	x_data = x_data.reshape(x_data.shape + (1,))
	# else:
	# 	x_data = x_data.reshape(x_data.shape + (1,1,))

	return x_data

def prepare_data_lstm_signal(x_data, args):
	"""Prepare data for lstm architecture.

	Parameters:
	x_train -- data like numpy array. 

	Return:
	Array of shape (samples, timesteps, len_vector)

	"""

	if x_data.ndim == 2:
		print("entre 2d")
		x_data = np.reshape(x_data, (x_data.shape[0], 1, x_data.shape[1]))

	elif x_data.ndim == 3:
		print("entre 3d")
		x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[2], x_data.shape[1]))

	else: #x_data.ndim == 4:
		x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[2], x_data.shape[1], 2))
		x_data = np.subtract(x_data[:,:,:,0], x_data[:,:,:,1])
	#for lstm timesteps is the second number in shape, i.e samples, timesteps, len_vector
	# if args.timesteps: #True
	# 	x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[2], x_data.shape[1])) 
	# else: #False
	# 	x_data = np.reshape(x_data, (x_data.shape[0], 1, x_data.shape[1]))
	
	return x_data


def prepare_data_cnn(x_data, args):
	"""Prepare data for 2D-CNN architecture.

	This functions return a 4D array for train a CNN architecture.
	Channels is the last element in the 4D array.

	Parameters:
	x_train -- data to reshape.

	Return:
	4D array of (samples, rows, cols, channels) this is the default of Keras implementation.

	"""

	if x_data.ndim == 2:
		x_data = x_data.reshape(x_data.shape + (1,1,))
		
	elif x_data.ndim == 3:
		x_data = x_data.reshape(x_data.shape + (1,))

	else: #4 dim
		x_data = x_data

	# if args.timesteps:
	# 	x_data = x_data.reshape(x_data.shape + (1,))
	# else:
	# 	x_data = x_data.reshape(x_data.shape + (1,1,))

	return x_data

def prepare_data_lstm(x_data, args):
	"""Prepare data for lstm architecture.

	Parameters:
	x_train -- data like numpy array. 

	Return:
	Array of shape (samples, timesteps, len_vector)

	"""

	if x_data.ndim == 2:
		print("entre 2d")
		x_data = np.reshape(x_data, (x_data.shape[0], 1, x_data.shape[1]))

	elif x_data.ndim == 3:
		print("entre 3d")
		x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[2], x_data.shape[1]))

	else: #x_data.ndim == 4:
		x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[2], x_data.shape[1], 2))
		x_data = np.subtract(x_data[:,:,:,0], x_data[:,:,:,1])
	#for lstm timesteps is the second number in shape, i.e samples, timesteps, len_vector
	# if args.timesteps: #True
	# 	x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[2], x_data.shape[1])) 
	# else: #False
	# 	x_data = np.reshape(x_data, (x_data.shape[0], 1, x_data.shape[1]))
	




















