from abc import ABC, abstractmethod

class InverseFoldModel(ABC):
	"""
	Base inverse folding model.
	"""

	@abstractmethod
	def predict(self, pdb_filepath):
		"""
		Predict sequences given an input pdb filepath.

		Args:
			pdb_filepath:
				PDB filepath of input structure.

		Returns:
			lines:
				Predicted sequences with statistics in FASTA format.
		"""
		raise NotImplemented