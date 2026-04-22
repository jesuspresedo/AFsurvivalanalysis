###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection, metrics

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils

from lib.rnn_baselines import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver

from lib.utils import compute_loss_all_batches

import sklearn
from confidenceinterval import roc_auc_score, recall_score, f1_score
from confidenceinterval.bootstrap import bootstrap_ci
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from confidenceinterval import precision_score

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")
parser.add_argument('--tau', type=int, default=730, help="Days to take into account before reference ECG for prediction")

args = parser.parse_args()

args = parser.parse_args(['--load', '5701', '--niters', '200', '-n', '8956', '-l', '15', '-b', '16', '--lr', '0.001', '--dataset', 'atrialfibrillation', '--latent-ode', '--rec-dims', '40', '--rec-layers', '3', '--gen-layers', '3', '--units', '30', '--gru-units', '100', '--extrap', '--classif'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

	start = time.time()
	print("Sampling dataset of {} training examples".format(args.n))
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, device)
	input_dim = data_obj["input_dim"]

	classif_per_tp = False
	if ("classif_per_tp" in data_obj):
		# do classification per time point rather than on a time series as a whole
		classif_per_tp = data_obj["classif_per_tp"]


	n_labels = 1
	if args.classif:
		if ("n_labels" in data_obj):
			n_labels = data_obj["n_labels"]
		else:
			raise Exception("Please provide number of labels for classification task")

	##################################################################
	# Create the model
	obsrv_std = 0.01

	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))


	if args.latent_ode:
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)


	#Load checkpoint and evaluate the model
	if args.load is not None:
		utils.get_ckpt_model(ckpt_path, model, device)
		all_test_labels = np.array([])
		classif_predictions = np.array([])
		for i in range(data_obj["n_test_batches"]):
			print(i)
			batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
			tp_to_predict = batch_dict['tp_to_predict']
			necgs = len(tp_to_predict)
			
			pred_x, info = model.get_reconstruction(tp_to_predict, batch_dict['observed_data'],batch_dict['observed_tp'],mask=batch_dict['observed_mask'],run_backwards=False,n_traj_samples=10)

			pred = info['label_predictions'].cpu().detach().numpy()
			
			if i == 0:
			    classif_predictions = pred[:,0,0,0].reshape(10,1)
			else:
			    classif_predictions = np.concatenate((classif_predictions,pred[:,0,0,0].reshape(10,1)),axis=1)
				
			labels = batch_dict['labels']
			labels = labels.squeeze(0)
			labels = labels.squeeze(1)
			
			necgs=1
			
			for ecg in range(necgs):
        			if labels[ecg]>0.5:
		        		all_test_labels = np.append(all_test_labels,1)
		        	else:
		        		all_test_labels = np.append(all_test_labels,0)
		auc1 = np.zeros(10)
		for i in range(10):
		    auc1[i] = metrics.roc_auc_score(all_test_labels,classif_predictions[i,:])
		    
		#print("mean AUC = ", np.mean(auc1))
		#print("std AUC = ", np.std(auc1))
		
		labels = []

		for i in range(10):
				for j in range(1792):
						labels.append(all_test_labels[j])
        
		predictions = []
		for i in range(10):
				for j in range(1792):
						predictions.append(classif_predictions[i,j])
                
		labels1 = [int(x) for x in labels]
		auc1,ci = roc_auc_score(labels1,predictions, confidence_level=0.95)#, method='bootstrap_bca', n_resamples=5000)
		print("AUC = ", auc1)
		print(ci)
		
		#precision, recall, thresholds = precision_recall_curve(labels,predictions)
		#pr_auc = auc(recall, precision)
		#print("pr AUC = ", pr_auc)
		#random_generator = np.random.default_rng()
		pr_auc, ci = bootstrap_ci(y_true=labels, y_pred=predictions, metric=average_precision_score, confidence_level=0.95, n_resamples=3000, method='bootstrap_bca')
		print("pr AUC = ", pr_auc)
		print(ci)
		

		binary_predictions = (np.array(predictions) >= -4.373883247375488).astype(int)
		tn, fp, fn, tp = confusion_matrix(labels, binary_predictions).ravel()
		sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall)
		sensitivity_ci = proportion_confint(tp, tp + fn, method='beta') if (tp + fn) > 0 else (0, 0)
		print("Sensitivity = ", sensitivity)
		print(sensitivity_ci)
		
		specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
		specificity_ci = proportion_confint(tn, tn + fp, method='beta') if (tn + fp) > 0 else (0, 0)
		print("Specificity = ", specificity)
		print(specificity_ci)
		
		accuracy = (tp + tn) / (tp + tn + fp + fn)  # Accuracy
		accuracy_ci = proportion_confint(tp + tn, tp + tn + fp + fn, method='beta')
		print("Accuracy = ", accuracy)
		print(accuracy_ci)
		
		f1, f1_ci = f1_score(labels, binary_predictions, confidence_level=0.95, average='binary', method='bootstrap_bca', n_resamples=3000)
		print("F1 = ", f1)
		print(f1_ci)
		
        
        
		exit()
