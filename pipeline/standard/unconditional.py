import os
import shutil

from pipeline.standard.base import Pipeline


class UnconditionalPipeline(Pipeline):
	"""
	Standard evaluation pipeline on unconditional generation outputs. Evaluation
	process consists of:
		-	self-consistency assessment on designability
		-	assessment on secondary structure diversity
	"""

	def evaluate(self, rootdir, clean=True, verbose=True):
		"""
		Evaluate a set of generated structures. Outputs are stored in the root directory,
		consisting of
			- 	A file named 'info.csv', which contains aggregated evaluation statistics 
				for the set of generated structures.
			-	A directory named 'designs', where each file is the most similar structure 
				(predicted by the folding model) to the generated structure and is stored 
				in a PDB format.

		Args:
			rootdir:
				Root directory containing a subdirectory named 'pdbs', where each
				file contains a generated structure in the PDB format
			clean:
				Whether to remove intermediate files and directories. Default to True.
			verbose:
				Whether to print detailed progress information. Default to True.
		"""

		##################
		###   Set up   ###
		##################

		assert os.path.exists(rootdir), 'Missing root directory'
		pdbs_dir = os.path.join(rootdir, 'pdbs')
		assert os.path.exists(pdbs_dir), 'Missing pdb directory'
		output_dir = rootdir

		###################
		###   Process   ###
		###################

		sequences_dir = self._inverse_fold(pdbs_dir, output_dir, verbose)
		structures_dir = self._fold(sequences_dir, output_dir, verbose)
		scores_dir = self._compute_scores(pdbs_dir, structures_dir, output_dir, verbose)
		results_dir, designs_dir = self._aggregate_scores(scores_dir, structures_dir, output_dir, verbose)
		self._compute_secondary_diversity(pdbs_dir, designs_dir, results_dir, verbose)
		self._process_results(results_dir, output_dir)

		####################
		###   Clean up   ###
		####################

		if clean:
			shutil.rmtree(sequences_dir)
			shutil.rmtree(structures_dir)
			shutil.rmtree(scores_dir)
			shutil.rmtree(results_dir)
