# DataSet of Atrial Fibrillation
# Author: Jesus Presedo

import torch
import glob
import math
import pandas as pd
import numpy as np
import lib.utils as utils
from torch.utils.data import DataLoader

# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max

class AtrialFibrillation(object):
	params = ['age', 'sex', 'qrsfrontaxis', 'meanqrsdur_cross', 'transqrsmaxmag', 'qdur_I', 'qrsdur_I', 'ramp_II', 'qrsarea_aVL', 'qrsdur_V5', 'qrsppk_V6', 'qrsarea_V6']

	params_dict = {k: i for i, k in enumerate(params)}

	labels = [ "FA" ]
	labels_dict = {k: i for i, k in enumerate(labels)}
	
	def __init__(self, args, download=False, generate=False, n_samples = None, device = torch.device("cpu")):
		
		if generate:
			self._generate_dataset(args, device)
			
	def _generate_dataset(self, args, device):
		total_dataset = []
		features = [0, 8, 22, 30, 59, 70, 97, 220, 439, 474, 476]
        
		files = glob.glob("../HospitalPrincesaAFDB/*.csv")
		f = pd.read_csv(files[0])
		columnas = f.columns
        
		columnas = columnas[features]
		
		files = sorted(files)
		for filename in files:
			record_id = filename.split('/')[-1].split('.')[0]
			print(record_id)
			f = pd.read_csv(filename)
            
			labels = f['FA']
			tt = f['tiempo']
			longitud = tt.size
	
			#for i in range(longitud):
			#	if labels[i] == 1:
			#		break
			#tt = tt[:i+1]
			tt = torch.tensor(tt).to(device)
			labels = torch.tensor(np.array(labels)).type(torch.float32).to(device)
			
			vals = np.array(f[columnas])
			#vals = vals[:i+1]
			mask = np.ones(vals.shape)
			fil = vals.shape[0]
			col = vals.shape[1]
			for i in range(fil):
				for j in range(col):
					if math.isnan(vals[i,j]):
						mask[i,j] = 0
						vals[i,j] = 0
			vals = torch.tensor(vals).type(torch.float32).to(device)
			mask = torch.tensor(mask).type(torch.float32).to(device)
			total_dataset.append((record_id, tt, vals, mask, labels))
		self.data = total_dataset
		
		

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def get_label(self, record_id):
		return self.labels[record_id]
    
def variable_time_collate_fn_atrialfib(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	if data_type == "train":
		if torch.round(batch[1][1][0]) == batch[1][1][0]:
			n_pac = len(batch)
			for i in range(n_pac):
				for j in range(len(batch[i][1])):
					batch[i][1][j] = batch[i][1][j] + i/n_pac
		

	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	
	combined_labels = None
	N_labels = 1

	#combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	combined_labels = torch.zeros([len(batch), len(combined_tt), N_labels]) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device = device)
	
	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		if labels is not None:
			labels = labels.to(device)

		indices = inverse_indices[offset:offset + len(tt)]
		offset += len(tt)

		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		if labels is not None:
			combined_labels[b, indices] = labels.unsqueeze(1)

	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
		att_min = data_min, att_max = data_max)

	if torch.max(combined_tt) != 0.:
		combined_tt = combined_tt / 4500
		
	data_dict = {
		"data": combined_vals, 
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
	return data_dict

if __name__ == '__main__':
	torch.manual_seed(1991)

	dataset = AtrialFibrillation(generate=True)
	dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn_atrialfib)
	print(dataloader.__iter__().next())
