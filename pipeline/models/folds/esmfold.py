import esm
import torch
import numpy as np

from pipeline.models.folds.base import FoldModel


class ESMFold(FoldModel):
	"""
	ESMFold structure prediction model.
	"""

	def __init__(self, device='cuda:0'):
		"""
		Args:
			device:
				Device name. Default to 'cuda:0'.
		"""
		self.model = esm.pretrained.esmfold_v1()
		self.model = self.model.eval().to(device)

	def predict(self, seq):
		"""
		Predict structure given an input sequence.

		Args:
			seq:
				Input sequence of length N.

		Returns:
			pdb_str:
				Predicted structure in a PDB format.
			pae:
				[N, N] Predicted Aligned Error matrix.
		"""
		with torch.no_grad():
			output = self.model.infer(seq, num_recycles=3)
			pdb_str = self.model.output_to_pdb(output)[0]
			pae = (output['aligned_confidence_probs'].cpu().numpy()[0] * np.arange(64)).mean(-1) * 31
			mask = output['atom37_atom_exists'].cpu().numpy()[0,:,1] == 1
		return pdb_str, pae[mask,:][:,mask]
