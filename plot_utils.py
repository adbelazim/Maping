import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from sklearn import manifold

import os

def get_target_names_dr(nb_classes, mode, args, variant_mode = "variantup"):
	"""Return a dictionary with target names for different cases of dataset.

	Parameters.
	mode -- Refers to the type of simulated data. All, Variant or Invariant.

	Return:
	A dictionary with target names and target colors for plotting purposes.
	"""

	assert nb_classes != 0

	if mode == "Invariant":
		#print("target names for Invariant data")
		if nb_classes == 10:
			target_names = {0:'ARI 0', 1:'ARI 1', 2:'ARI 2', 3:'ARI 3', 4:'ARI 4', 5:'ARI 5', \
							6:'ARI 6', 7: 'ARI 7', 8: 'ARI 8', 9: 'ARI 9', 10: str(args.subject)}
			target_colors = {0:'#000000', 1:'#800000', 2:'#FF0000', 3:'#FF5733', 4:'#FFFF00', 5:'#00FFFF', \
							6:'#008080', 7: '#0000FF', 8: '#008000', 9: '#00FF00', 10: '#ff0065'}

		elif nb_classes == 5:
			target_names = {0:'ARI 0-1', 1:'ARI 2-3', 2:'ARI 4-5', 3:'ARI 6-7', 4:'ARI 8-9', 10: str(args.subject)}
			target_colors = {0:'#800000', 1:'#FF0000', 2:'#FFFF00', 3:'#0000FF', 4:'#00FF00', 10: '#ff0065'}

		elif nb_classes == 3:
			target_names = {0:'ARI 1-2-3', 1:'ARI 4-5-6', 2:'ARI 7-8-9', 10: str(args.subject)}
			target_colors = {0:'#FF0000', 1:'#0000FF', 2:'#008000', 10: '#ff0065'}

		else: #nb_classes = 2
			target_names = {0:'ARI 0-1-2-3-4', 1:'ARI 5-6-7-8-9', 10: str(args.subject)}
			target_colors = {0:'#008080', 1:'#00FF00', 10: '#ff0065'}

	elif mode == "Variant":
		#print("target names for Variant data")
		if nb_classes == 2:
			target_names = {0:'ARI 1->9', 1:'ARI 9->1', 10: str(args.subject)}
			target_colors = {0:'#00FF00', 1:'#FF0000', 10: '#ffc966'}

		elif nb_classes == 3 and variant_mode == "variantdown":
			target_names = {0:'ARI 3->1', 1:'ARI 6->4', 2:'ARI 9->7', 10: str(args.subject)}
			target_colors = {0:'#FF0000', 1:'#0000FF', 2:'#00FF00', 10: '#ffc966'}

		elif nb_classes == 3 and variant_mode == "variantup":
			target_names = {0:'ARI 1->3', 1:'ARI 4->6', 2:'ARI 7->9', 10: str(args.subject)}
			target_colors = {0:'#FF0000', 1:'#0000FF', 2:'#00FF00', 10: '#ffc966'}


		else: #nb_classes = 4 and variant mode = variantbi
			target_names = {0:'ARI 0->4', 1:'ARI 4->0', 2:'ARI 5->9', 3:'ARI 9->5', 10: str(args.subject)}
			target_colors = {0:'#00FFFF', 1:'#FF0000', 2:'#00FF00', 3:'#0000FF', 10: '#ffc966'}

	else: #mode "All"
		#print("target names for All data")
		if nb_classes == 7:
			target_names = {0:'ARI 1-2-3', 1: 'ARI 4-5-6', 2: 'ARI 7-8-9', 3: 'ARI 0->4', \
							4: 'ARI 4->0', 5: 'ARI 5->9', 6: 'ARI 9->5', 10: str(args.subject)}
			target_colors = {0:'#FF0000', 1: '#0000FF', 2: '#008000', 3: '#008080', \
							4: '#FFFF00', 5: '#00FF00', 6: '#FF00FF', 10: '#ff0065'}

		elif nb_classes == 4 and variant_mode == 'variant22':
			target_names = {0: 'ARI 0-1-2-3-4', 1: 'ARI 5-6-7-8-9', 2: 'ARI 1->9', 3: 'ARI 9->1', 10: str(args.subject)}
			target_colors = {0: '#008080', 1: '#00FF00', 2: '#00FFFF', 3: '#FFFF00', 10: '#ff0065'}

		else: 
			target_names = {0: 'ARI 0-1-2-3-4', 1: 'ARI 5-6-7-8-9', 2: 'ARI 0->4 and 5->9', 3: 'ARI 9->5 and 4->0', 10: str(args.subject)} 
			target_colors = {0: '#008080', 1: '#00FF00', 2: '#0000FF', 3: '#FFFF00', 10: '#ff0065'} 

	return target_names, target_colors


def tsne_2d(data_matrix, labels):
	"""Retorna en un pandas dataframe TSNE de dos dimensiones.

	El dataframe contiene x,y, label.

	"""
	print("tsne 2d....")
	tsne = manifold.TSNE(n_components=2, init='pca', method="exact",random_state=0)
	pos = tsne.fit_transform(data_matrix)
	xs, ys = pos[:, 0], pos[:, 1]
	df = pd.DataFrame(dict(x=xs, y=ys, label=labels))
	return df 


def plot_visualization(path_results, x_data, y_data, variant_mode, nb_classes, signal_test, args):
	"""Plot the dimensionality_reduction method (t-sne).

	Parameters:
	data_frame -- 
	filename -- 
	args --


	"""

	#path_tsne = path_results + "/Visualization/train/" + str(args.step) + "_2d.csv"
	#data_frame = pd.read_csv(path_tsne)
	
	path_maping = path_results + "/Maping/" + str(args.subject).split(".txt")[0] + "/"
	filename = path_maping + "maping_" + str(args.step) + "_" + str(args.subject).split(".txt")[0] + "_stick" + str(args.stick) + ".png"

	print("path_save maping", path_maping)

	if not os.path.exists(path_maping):
		os.makedirs(path_maping)

	#print("path_tsne", path_tsne)

	label_maping = np.array([10])

	x_data = np.concatenate((x_data,signal_test),axis=0)
	y_data = np.concatenate((y_data,label_maping),axis=0)

	print("x_data concatenate",x_data.shape)
	print("y_data concatenate",y_data.shape)

	data_frame = tsne_2d(x_data, y_data)

	
	
	groups = data_frame.groupby('label')

	cluster_names, cluster_colors = get_target_names_dr(nb_classes, args.mode, args, variant_mode)

	fig = plt.figure(figsize=(20, 10))
	ax = fig.add_subplot(111)
	ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	for name, group in groups:
		
		if cluster_names[name] == str(args.subject):
			ax.scatter(group.x, group.y, marker='D', s=150, edgecolors = 'face',label=cluster_names[name], color=cluster_colors[name])
		else:
			ax.scatter(group.x, group.y, marker='o', label=cluster_names[name], color=cluster_colors[name])

	ax.legend(numpoints=1)  #show legend with only 1 point
	plt.savefig(filename) #save the plot












































