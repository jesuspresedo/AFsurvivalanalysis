###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from lib.diffeq_solver import DiffeqSolver
from torch.distributions import uniform
from torch.utils.data import DataLoader
from atrial_fibrillation import AtrialFibrillation, variable_time_collate_fn_atrialfib, get_data_min_max



from sklearn import model_selection
import random

#####################################################################################################
def parse_datasets(args, device):
	

	def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
		batch = torch.stack(batch)
		data_dict = {
			"data": batch, 
			"time_steps": time_steps}

		data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
		return data_dict


	dataset_name = args.dataset

	n_total_tp = args.timepoints + args.extrap
	max_t_extrap = args.max_t / args.timepoints * n_total_tp





	##################################################################
	# Atrial fibrillation dataset
	if dataset_name == 'atrialfibrillation':
		total_dataset = AtrialFibrillation(args=args, generate=True, device=device)
        
		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, 
			random_state = 42, shuffle = True)
        
		record_id, tt, vals, mask, labels = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		batch_size = min(min(len(total_dataset), args.batch_size), args.n)
		
		# Normalizamos todo el conjunto de datos (train + test)
		#data_min, data_max = get_data_min_max(total_dataset)
		data_min = torch.tensor([ 1.0000e+00, -9.0000e+01,  4.4000e+01,  3.5000e+01,  0.0000e+00, 0.0000e+00, -3.4100e+02, -1.1620e+03,  0.0000e+00,  0.0000e+00, -1.2780e+03], device='cuda:0')
		data_max = torch.tensor([ 105.,  269.,  264., 4957.,  260.,  336., 3991., 1236.,  245., 7431., 617.], device='cuda:0')

		train_dataloader = DataLoader(train_data, batch_size = 1, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_atrialfib(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max))



		test_dataloader = DataLoader(test_data, batch_size = 1, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_atrialfib(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))

		attr_names = total_dataset.params
		data_objects = {"dataset_obj": total_dataset, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"attr": attr_names, #optional
					"classif_per_tp": True, #False, #optional
					"n_labels": 1} #optional
		return data_objects
