import os

import utils

def generate_invariant_data(path_data, args):

	subjects = os.listdir(path_data)
	print("subjects",subjects)

	for subject in subjects:
		if subject == args.subject:
			#DY subject
			cbfv_signal = utils.read_cbfv_file(path_data + "/" +subjects[2])
			#cambia frecuencia de muestreo a 0.6
			cbfv_signal = utils.lineal_interpolation(cbfv_signal)



def generate_variant_data(path_data_nc, path_data_hc):

	return 0

def main(path_env):

	path_data_nc = path_env + "/nc_data/"
	path_data_hc = path_env + "/hc_data/"

	generate_invariant_data(path_data_nc)



if __name__ == "__main__":
	#example execution
	#python maping.py -m Variant -c Mix -train_data abp_cbfv_nc_4class_variantbi_freq0.6 -val_data abp_cbfv_hc_4class_variantbi_freq0.6 -model resnet -f 1 -step 500 -subject DY0000.txt

	path = os.getcwd()
	path_env, path_source = os.path.split(path)

	main(path)