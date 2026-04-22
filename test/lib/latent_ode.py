###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import sklearn as sk
import numpy as np
#import gc
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
from lib.base_models import VAE_Baseline



class LatentODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
		z0_prior, device, obsrv_std = None, 
		use_binary_classif = False,
		linear_classifier = False,
		classif_per_tp = False,
		n_labels = 1,
		train_classif_w_reconstr = False):

		super(LatentODE, self).__init__(
			input_dim = input_dim, latent_dim = latent_dim, 
			z0_prior = z0_prior, 
			device = device, obsrv_std = obsrv_std, 
			use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp, 
			linear_classifier = linear_classifier,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder

	def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
		mask = None, n_traj_samples = 1, run_backwards = True, mode = None):

		if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
			isinstance(self.encoder_z0, Encoder_z0_RNN):

			truth_w_mask = truth
			if mask is not None:
				truth_w_mask = torch.cat((truth, mask), -1)
			first_point_mu, first_point_std = self.encoder_z0(
				truth_w_mask, truth_time_steps, run_backwards = run_backwards)

			means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
			sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
			first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

		else:
			raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
		
		first_point_std = first_point_std.abs()
		assert(torch.sum(first_point_std < 0) == 0.)

		first_point_enc_aug = first_point_enc
		means_z0_aug = means_z0
			
		assert(not torch.isnan(time_steps_to_predict).any())
		assert(not torch.isnan(first_point_enc).any())
		assert(not torch.isnan(first_point_enc_aug).any())

# Resuelve la ecuación diferencial ordinaria para n_traj_samples trayectorias, n_samples pacientes, n_timepoints puntos de muestreo
# y n_latents número de variables latentes
		# Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

		pred_x = self.decoder(sol_y)

		all_extra_info = {
			"first_point": (first_point_mu, first_point_std, first_point_enc),
			"latent_traj": sol_y.detach()
		}

		if self.use_binary_classif:
			if self.classif_per_tp:
				all_extra_info["label_predictions"] = self.classifier(sol_y)
			else:
				all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

		return pred_x, all_extra_info


	def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
		# input_dim = starting_point.size()[-1]
		# starting_point = starting_point.view(1,1,input_dim)

		# Sample z0 from prior
		starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

		starting_point_enc_aug = starting_point_enc

		sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict, 
			n_traj_samples = 3)

		return self.decoder(sol_y)
