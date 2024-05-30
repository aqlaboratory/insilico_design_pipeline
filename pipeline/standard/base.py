import os
import glob
import torch
import shutil
import subprocess
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from abc import ABC, abstractmethod

from pipeline.utils.cluster import hcluster
from pipeline.utils.parse import (
	parse_pdb_file,
	parse_tm_file,
	parse_pae_file
)
from pipeline.utils.secondary import (
	assign_secondary_structures,
	assign_left_handed_helices
)


class Pipeline(ABC):
	"""
	Base standard evaluation pipeline. The standard pipeline consists of self-consistency
	assessment on designability and secondary structure evaluations.

	NOTE: Current secondary structure evaluation uses the P-SEA algorithm, which predicts
	secondary structure elements based on Ca atom coordinates.
	"""

	def __init__(
		self,
		inverse_fold_model,
		fold_model,
		tm_score_exec='packages/TMscore/TMscore',
		tm_align_exec='packages/TMscore/TMalign'
	):
		"""
		Args:
			inverse_fold_model:
				Inverse folding model (an instance of a derived class whose base class
				is defined in pipeline/models/inverse_folds/base.py).
			fold_model:
				Structure prediction model (an instance of a derived class whose base
				class is defined in pipeline/models/folds/base.py).
			tm_score_exec:
				Path to TMscore executable. Default to 'packages/TMscore/TMscore'.
			tm_align_exec:
				Path to TMalign executable. Default to 'packages/TMalign/TMalign'.
		"""
		self.inverse_fold_model = inverse_fold_model
		self.fold_model = fold_model
		self.tm_score_exec = tm_score_exec
		self.tm_align_exec = tm_align_exec

	@abstractmethod
	def evaluate(self, rootdir, clean=True, verbose=True):
		"""
		Main evaluation function.

		Args:
			rootdir:
				Root directory.
			clean:
				Whether to remove intermediate files and directories. Default to True.
			verbose:
				Whether to print detailed progress information. Default to True.
		"""
		raise NotImplemented

	def _inverse_fold(self, pdbs_dir, output_dir, verbose):
		"""
		Run inverse folding to obtain sequences.

		Args:
			pdbs_dir:
				Directory containing PDB files for generated structures.
			output_dir:
				Base output directory.
			verbose:
				Whether to print detailed progress information.

		Returns:
			sequences_dir:
				Output directory (specified as [output_dir]/sequences), where each file
				contains predicted sequences and their corresponding statistics in a FASTA
				format for the generated structure.
		"""

		# Create output directory
		sequences_dir = os.path.join(output_dir, 'sequences')
		assert not os.path.exists(sequences_dir), 'Output sequences directory existed'
		os.mkdir(sequences_dir)

		# Process
		for pdb_filepath in tqdm(
			glob.glob(os.path.join(pdbs_dir, '*.pdb')),
			desc='Inverse folding', disable=not verbose
		):
			domain_name = pdb_filepath.split('/')[-1].split('.')[0]
			sequences_filepath = os.path.join(sequences_dir, f'{domain_name}.txt')
			with open(sequences_filepath, 'w') as file:
				file.write(self.inverse_fold_model.predict(pdb_filepath))

		return sequences_dir

	def _fold(self, sequences_dir, output_dir, verbose):
		"""
		Run folding to obtain structures.

		Args:
			sequences_dir:
				Sequence directory where each file contains predicted sequences and their
				corresponding statistics in a FASTA format for a generated structure.
			output_dir:
				Base output directory.
			verbose:
				Whether to print detailed progress information.

		Returns:
			structures_dir:
				Output directory (specified as [output_dir]/structures), where each .pdb file 
				contains the predicted structue in a PDB format and each .pae.txt file contains 
				the predicted Aligned Error (pAE) matrix.
		"""

		# Create output directory
		structures_dir = os.path.join(output_dir, 'structures')
		assert not os.path.exists(structures_dir), 'Output structures directory existed'
		os.mkdir(structures_dir)

		# Process
		for filepath in tqdm(
			glob.glob(os.path.join(sequences_dir, '*.txt')),
			desc='Folding', disable=not verbose
		):
			domain_name = filepath.split('/')[-1].split('.')[0]
			with open(filepath) as file:
				seqs = [line.strip() for line in file if line[0] != '>']
			for i in range(len(seqs)):

				# Define output filepaths
				output_pdb_filepath = os.path.join(structures_dir, f'{domain_name}-resample_{i}.pdb')
				output_pae_filepath = os.path.join(structures_dir, f'{domain_name}-resample_{i}.pae.txt')
				
				# Run structure prediction
				pdb_str, pae = self.fold_model.predict(seqs[i])
				
				# Save
				np.savetxt(output_pae_filepath, pae, '%.3f')
				with open(output_pdb_filepath, 'w') as f:
					f.write(pdb_str)

		return structures_dir

	def _compute_scores(self, pdbs_dir, structures_dir, output_dir, verbose):
		"""
		Compute self-consistency scores.

		Args:
			pdbs_dir:
				Directory containing PDB files for generated structures.
			structures_dir:
				Directory containing details on structures predicted by folding model,
				where each .pdb file contains the predicted structue in a PDB format and
				each .pae.txt file contains the predicted Aligned Error (pAE) matrix.
			output_dir:
				Base output directory.
			verbose:
				Whether to print detailed progress information.

		Returns:
			scores_dir:
				Output directory (specified as [output_dir]/scores), where each file 
				contains the ouput from running TMscore on a structure predicted by 
				the folding (structure prediction) model.
		"""

		# Create output directory
		scores_dir = os.path.join(output_dir, 'scores')
		assert not os.path.exists(scores_dir), 'Output scores directory existed'
		os.mkdir(scores_dir)

		# Process
		for designed_pdb_filepath in tqdm(
			glob.glob(os.path.join(structures_dir, '*.pdb')),
			desc='Computing scores', disable=not verbose
		):

			# Parse
			filename = designed_pdb_filepath.split('/')[-1].split('.')[0]
			domain_name = '-'.join(filename.split('-')[:-1])
			seq_name = filename.split('-')[-1]

			# Compute score
			generated_pdb_filepath = os.path.join(pdbs_dir, f"{domain_name}.pdb")
			output_filepath = os.path.join(scores_dir, f'{domain_name}-{seq_name}.txt')
			subprocess.call(f'{self.tm_score_exec} {generated_pdb_filepath} {designed_pdb_filepath} > {output_filepath}', shell=True)

		return scores_dir

	def _aggregate_scores(self, scores_dir, structures_dir, output_dir, verbose):
		"""
		Aggregate self-consistency scores and structural confidence scores.
		Save best resampled structures.

		Args:
			scores_dir:
				Score directory where each file contains the ouput from running TMscore 
				on a structure predicted by the folding (structure prediction) model.
			structures_dir:
				Directory containing details on structures predicted by folding model,
				where each .pdb file contains the predicted structue in a PDB format and
				each .pae.txt file contains the predicted Aligned Error (pAE) matrix.
			output_dir:
				Base output directory.
			verbose:
				Whether to print detailed progress information.

		Returns:
			results_dir:
				Result directory containing a file named 'single_scores.csv', where 
				each line stores the self-consistency evaluation results on a generated
				structure.
			designs_dir:
				Directory where each file is the most similar structure (predicted by the
				folding model) to the generated structure and is stored in a PDB format.
		"""

		# Create output directory
		results_dir = os.path.join(output_dir, 'results')
		designs_dir = os.path.join(output_dir, 'designs')
		assert not os.path.exists(results_dir), 'Output results directory existed'
		assert not os.path.exists(designs_dir), 'Output designs directory existed'
		os.mkdir(results_dir)
		os.mkdir(designs_dir)

		# Create scores filepath
		scores_filepath = os.path.join(results_dir, 'single_scores.csv')
		with open(scores_filepath, 'w') as file:
			columns = ['domain', 'seqlen', 'scTM', 'scRMSD', 'pLDDT', 'pAE']
			file.write(','.join(columns) + '\n')

		# Get domains
		domains = set()
		for filepath in glob.glob(os.path.join(scores_dir, '*.txt')):
			domains.add('-'.join(filepath.split('/')[-1].split('-')[:-1]))
		domains = list(domains)

		# Process
		for domain in tqdm(domains, desc='Aggregating scores', disable=not verbose):

			# Find best sample based on scRMSD
			resample_idxs, scrmsds = [], []
			for filepath in glob.glob(os.path.join(scores_dir, f'{domain}-resample_*.txt')):
				resample_idx = int(filepath.split('_')[-1].split('.')[0])
				resample_results = parse_tm_file(filepath)
				resample_idxs.append(resample_idx)
				scrmsds.append(resample_results['rmsd'])
			best_resample_idx = resample_idxs[np.argmin(scrmsds)]

			# Parse scores
			tm_filepath = os.path.join(
				scores_dir,
				f'{domain}-resample_{best_resample_idx}.txt'
			)
			output = parse_tm_file(tm_filepath)
			sctm, scrmsd, seqlen = output['tm'], output['rmsd'], output['seqlen']

			# Parse pLDDT
			pdb_filepath = os.path.join(
				structures_dir,
				f'{domain}-resample_{best_resample_idx}.pdb'
			)
			output = parse_pdb_file(pdb_filepath)
			plddt = np.mean(output['pLDDT'])

			# Parse pAE
			pae_filepath = os.path.join(
				structures_dir,
				f'{domain}-resample_{best_resample_idx}.pae.txt'
			)
			pae = parse_pae_file(pae_filepath)['pAE'] if os.path.exists(pae_filepath) else None

			# Save results
			with open(scores_filepath, 'a') as file:
				file.write('{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
					domain, seqlen, sctm, scrmsd, plddt, pae
				))

			# Save best resampled structure
			design_filepath = os.path.join(designs_dir, f'{domain}.pdb')
			shutil.copyfile(pdb_filepath, design_filepath)

		return results_dir, designs_dir

	def _compute_secondary_diversity(self, pdbs_dir, designs_dir, results_dir, verbose):
		"""
		Compute secondary diversity. Outputs are stored in the results directory, where each line 
		in the file provides secondary structure statistics on a generated structure or its most 
		similar structure predicted by the structure prediction model.

		Args:
			pdbs_dir:
				Directory containing PDB files for generated structures.
			designs_dir:
				Directory where each file is the most similar structure (predicted by the
				folding model) to the generated structure and is stored in a PDB format.
			results_dir:
				Result directory containing a file named 'single_scores.csv', where 
				each line stores the self-consistency evaluation results on a generated
				structure.
		"""

		# Create output filepath
		assert os.path.exists(results_dir), 'Missing output results directory'
		generated_secondary_filepath = os.path.join(results_dir, 'single_generated_secondary.csv')
		assert not os.path.exists(generated_secondary_filepath), 'Output generated secondary filepath existed'
		with open(generated_secondary_filepath, 'w') as file:
			columns = ['domain', 'generated_pct_helix', 'generated_pct_strand', 'generated_pct_ss', 'generated_pct_left_helix']
			file.write(','.join(columns) + '\n')
		designed_secondary_filepath = os.path.join(results_dir, 'single_designed_secondary.csv')
		assert not os.path.exists(designed_secondary_filepath), 'Output designed secondary filepath existed'
		with open(designed_secondary_filepath, 'w') as file:
			columns = ['domain', 'designed_pct_helix', 'designed_pct_strand', 'designed_pct_ss', 'designed_pct_left_helix']
			file.write(','.join(columns) + '\n')

		# Process generated pdbs
		for generated_filepath in tqdm(
			glob.glob(os.path.join(pdbs_dir, '*.pdb')),
			desc='Computing generated secondary diversity', disable=not verbose
		):

			# Parse filepath
			domain = generated_filepath.split('/')[-1].split('.')[0]

			# Parse pdb file
			output = parse_pdb_file(generated_filepath)

			# Parse secondary structures
			ca_coords = torch.Tensor(output['ca_coords']).unsqueeze(0)
			pct_ss = torch.sum(assign_secondary_structures(ca_coords, full=False), dim=1).squeeze(0) / ca_coords.shape[1]
			pct_left_helix = torch.sum(assign_left_handed_helices(ca_coords).squeeze(0)) / ca_coords.shape[1]

			# Save
			with open(generated_secondary_filepath, 'a') as file:
				file.write('{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
					domain, pct_ss[0], pct_ss[1], pct_ss[0] + pct_ss[1], pct_left_helix
				))

		# Process designed pdbs
		for design_filepath in tqdm(
			glob.glob(os.path.join(designs_dir, '*.pdb')),
			desc='Computing designed secondary diversity', disable=not verbose
		):

			# Parse filepath
			domain = design_filepath.split('/')[-1].split('.')[0]

			# Parse pdb file
			output = parse_pdb_file(design_filepath)

			# Parse secondary structures
			ca_coords = torch.Tensor(output['ca_coords']).unsqueeze(0)
			pct_ss = torch.sum(assign_secondary_structures(ca_coords, full=False), dim=1).squeeze(0) / ca_coords.shape[1]
			pct_left_helix = torch.sum(assign_left_handed_helices(ca_coords).squeeze(0)) / ca_coords.shape[1]

			# Save
			with open(designed_secondary_filepath, 'a') as file:
				file.write('{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
					domain, pct_ss[0], pct_ss[1], pct_ss[0] + pct_ss[1], pct_left_helix
				))

	def _process_results(self, results_dir, output_dir):
		"""
		Combine files in the results directory and output a file named 'info.csv' under the
		output directory, which contains aggregated evaluation statistics for the set of 
		generated structures.

		Args:
			results_dir:
				Result directory where each file contains aggregated information on the set
				of generated structure and each line in the file contains statistics on a
				generated structure.
			output_dir:
				Base output directory.
		"""

		# Create output filepath
		assert os.path.exists(results_dir), 'Missing output results directory'
		info_filepath = os.path.join(output_dir, 'info.csv')
		assert not os.path.exists(info_filepath), 'Output info filepath existed'

		# Process single level information
		for idx, filepath in enumerate(glob.glob(os.path.join(results_dir, 'single_*.csv'))):
			if idx == 0:
				df = pd.read_csv(filepath)
			else:
				df = df.merge(pd.read_csv(filepath), on='domain')

		# Save single level information
		df.to_csv(info_filepath, index=False)
