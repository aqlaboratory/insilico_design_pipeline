from abc import ABC, abstractmethod


class FoldModel(ABC):
	"""
	Base folding (structure prediction) model.
	"""

	@abstractmethod
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
		raise NotImplemented