import os
import glob
import torch
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

from pipeline.standard.base import Pipeline
from pipeline.utils.parse import parse_tm_file
from pipeline.utils.align import compute_rigid_alignment


class ScaffoldPipeline(Pipeline):
	"""
	Standard evaluation pipeline on motif scaffolding outputs. Evaluation
	process consists of:
		-	self-consistency assessment on designability
		-	assessment on secondary structure diversity
		-	assessment on motif constraint satisfaction.
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
				Root directory containing
					-	a subdirectory named 'pdbs', where each file contains a
						generated structure in the PDB format
					-	a subdirectory named 'motif_pdbs', where each corresponding
						file (same filename as the filename in the 'pdbs' subdirectory)
						contains the motif structure, aligned in residue indices with 
						the generated structure and stored in the PDB format.
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
		motif_pdbs_dir = os.path.join(rootdir, 'motif_pdbs')
		assert os.path.exists(pdbs_dir), 'Missing pdb directory'
		assert os.path.exists(motif_pdbs_dir), 'Missing motif pdb directory'
		output_dir = rootdir

		###################
		###   Process   ###
		###################

		processed_pdb_dir = self._map_motif_sequence(motif_pdbs_dir, pdbs_dir, output_dir, verbose)
		sequences_dir = self._inverse_fold_scaffold(motif_pdbs_dir, processed_pdb_dir, output_dir, verbose)
		structures_dir = self._fold(sequences_dir, output_dir, verbose)
		scores_dir = self._compute_scores(pdbs_dir, structures_dir, output_dir, verbose)
		results_dir, designs_dir = self._aggregate_scores(scores_dir, structures_dir, output_dir, verbose)
		self._compute_secondary_diversity(pdbs_dir, designs_dir, results_dir, verbose)
		self._compute_motif_scores(motif_pdbs_dir, designs_dir, results_dir, verbose)
		self._process_results(results_dir, output_dir)

		####################
		###   Clean up   ###
		####################

		if clean:
			shutil.rmtree(processed_pdb_dir)
			shutil.rmtree(sequences_dir)
			shutil.rmtree(structures_dir)
			shutil.rmtree(scores_dir)
			shutil.rmtree(results_dir)

	def _map_motif_sequence(self, motif_pdbs_dir, pdbs_dir, output_dir, verbose):
		"""
		Map motif sequence information into PDB files of generated structures 
		in preparation for later conditional inverse folding.

		Args:
			motif_pdbs_dir:
				Directory containing motif structures, where each PDB file (corresponding to 
				the same filename in the pdbs directory) contains the motif structure, aligned 
				in residue indices with the generated structure.
			pdbs_dir:
				Directory containing generated structures in the PDB format.
			output_dir:
				Base output directory.
			verbose:
				Whether to print detailed progress information.

		Returns:
			processed_pdb_dir:
				Output directory (specified as [output_dir]/processed_pdbs), where each file 
				contains the generated structure in the PDB format, with mapped motif sequence 
				information.
		"""

		# Create output directory
		processed_pdb_dir = os.path.join(output_dir, 'processed_pdbs')
		assert not os.path.exists(processed_pdb_dir), 'Output processed pdbs directory existed'
		os.mkdir(processed_pdb_dir)

		# Process
		for pdb_filepath in tqdm(
			glob.glob(os.path.join(pdbs_dir, '*.pdb')),
			desc='Mapping motif sequence', disable=not verbose
		):

			# Parse
			domain_name = pdb_filepath.split('/')[-1].split('.')[0]

			# Create residue index to name mapping
			motif_pdb_filepath = os.path.join(motif_pdbs_dir, f'{domain_name}.pdb')
			with open(motif_pdb_filepath) as file:
				residue_name_dict = dict([
					(int(line[22:26]), line[17:20]) for line in file
					if line.startswith('ATOM') and line[12:16].strip() == 'CA'
				])

			# Update
			lines = []
			with open(pdb_filepath) as file:
				for line in file:
					assert line.startswith('ATOM') and line[21] == 'A'
					residue_index = int(line[22:26])
					residue_name = line[17:20]
					if residue_index in residue_name_dict:
						residue_name = residue_name_dict[residue_index]
					lines.append(line[:17] + residue_name + line[20:])

			# Save
			processed_pdb_filepath = os.path.join(processed_pdb_dir, f'{domain_name}.pdb')
			with open(processed_pdb_filepath, 'w') as file:
				file.write(''.join(lines))

		return processed_pdb_dir

	def _inverse_fold_scaffold(self, motif_pdbs_dir, processed_pdbs_dir, output_dir, verbose):
		"""
		Run conditional inverse folding to obtain sequences.

		Args:
			motif_pdbs_dir:
				Directory containing motif structures, where each PDB file (corresponding to 
				the same filename in the pdbs directory) contains the motif structure, aligned 
				in residue indices with the generated structure.
			processed_pdb_dir:
				Directory containing processed PDB files, where each file contains the generated 
				structure in the PDB format, with mapped motif sequence information.
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
		for processed_pdb_filepath in tqdm(
			glob.glob(os.path.join(processed_pdbs_dir, '*.pdb')),
			desc='Inverse folding', disable=not verbose
		):
			domain_name = processed_pdb_filepath.split('/')[-1].split('.')[0]
			sequences_filepath = os.path.join(sequences_dir, f'{domain_name}.txt')

			# Create fixed positions dictionary
			with open(os.path.join(motif_pdbs_dir, f'{domain_name}.pdb')) as file:
				fixed_residue_indices = [
					int(line[22:26]) for line in file 
					if line.startswith('ATOM') and line[12:16].strip() == 'CA'
				]
			fixed_positions_dict = {}
			fixed_positions_dict[domain_name] = {
				'A': fixed_residue_indices
			}

			# Predict
			with open(sequences_filepath, 'w') as file:
				file.write(
					self.inverse_fold_model.predict(
						processed_pdb_filepath,
						fixed_positions_dict=fixed_positions_dict
					)
				)

		return sequences_dir
	
	def _compute_motif_scores(self, motif_pdbs_dir, designs_dir, results_dir, verbose):
		"""
		Compute statistics on motif constraint satisfactions. Outputs are stored in the 
		results directory, where each line in the file provides statistics on motif 
		constraint satisfactions for a generated structure.

		Args:
			motif_pdbs_dir:
				Directory containing motif structures, where each PDB file (corresponding 
				to the same filename in the pdbs directory) contains the motif structure,
				aligned in residue indices with the generated structure.
			designs_dir:
				Directory where each file is the most similar structure (predicted by the
				folding model) to the generated structure and is stored in a PDB format.
			results_dir:
				Result directory where each file contains aggregated information on the 
				set of generated structure and each line in the file contains statistics
				on a generated structure.
			verbose:
				Whether to print detailed progress information.
		"""

		# Create scores filepath
		motif_scores_filepath = os.path.join(results_dir, 'single_motif_scores.csv')
		with open(motif_scores_filepath, 'w') as file:
			columns = ['domain', 'motif_ca_rmsd', 'motif_bb_rmsd']
			file.write(','.join(columns) + '\n')

		# Process
		for design_pdb_filepath in tqdm(
			glob.glob(os.path.join(designs_dir, '*.pdb')),
			desc='Computing motif scores', disable=not verbose
		):

			# Parse
			name = design_pdb_filepath.split('/')[-1].split('.')[0]
			motif_pdb_filepath = os.path.join(motif_pdbs_dir, f'{name}.pdb')
			motif_groups = OrderedDict()
			residx_to_group = {}
			with open(motif_pdb_filepath) as file:
				for line in file:
					assert line.startswith('ATOM')
					group = line[72:76].strip()
					residx = int(line[22:26])

					# Create new group if necessary
					if group not in motif_groups:
						motif_groups[group] = {
							'ca_coords': [],
							'bb_coords': []
						}

					# Store coordinates
					coord = [
						float(line[30:38].strip()),
						float(line[38:46].strip()),
						float(line[46:54].strip())
					]
					if line[12:16].strip() in ['C', 'CA', 'N', 'O']:
						motif_groups[group]['bb_coords'].append(coord)
					if line[12:16].strip() == 'CA':
						motif_groups[group]['ca_coords'].append(coord)
						residx_to_group[residx] = group

			# Extract
			designed_motif_groups = OrderedDict()
			with open(design_pdb_filepath) as file:
				for line in file:
					if line.startswith('ATOM'):
						residx = int(line[22:26])
						if residx not in residx_to_group:
							continue

						# Create new group if necessary
						group = residx_to_group[residx]
						if group not in designed_motif_groups:
							assert group in motif_groups
							designed_motif_groups[group] = {
								'ca_coords': [],
								'bb_coords': []
							}

						# Store coordinates
						coord = [
							float(line[30:38].strip()),
							float(line[38:46].strip()),
							float(line[46:54].strip())
						]
						if line[12:16].strip() in ['C', 'CA', 'N', 'O']:
							designed_motif_groups[group]['bb_coords'].append(coord)
						if line[12:16].strip() == 'CA':
							designed_motif_groups[group]['ca_coords'].append(coord)

			# Iterate
			assert len(motif_groups) == len(designed_motif_groups)
			motif_bb_rmsds, motif_ca_rmsds = [], []
			for group in motif_groups:

				# Parse
				seg_motif_ca_coords = motif_groups[group]['ca_coords']
				seg_motif_bb_coords = motif_groups[group]['bb_coords']
				seg_designed_motif_ca_coords = designed_motif_groups[group]['ca_coords']
				seg_designed_motif_bb_coords = designed_motif_groups[group]['bb_coords']
				assert len(seg_motif_ca_coords) == len(seg_designed_motif_ca_coords)
				assert len(seg_motif_bb_coords) == len(seg_designed_motif_bb_coords)

				# Convert to tensor
				seg_motif_bb_coords = torch.Tensor(seg_motif_bb_coords)
				seg_motif_ca_coords = torch.Tensor(seg_motif_ca_coords)
				seg_designed_motif_bb_coords = torch.Tensor(seg_designed_motif_bb_coords)
				seg_designed_motif_ca_coords = torch.Tensor(seg_designed_motif_ca_coords)

				# Comptue motif backbone rmsd
				R, t = compute_rigid_alignment(
					seg_designed_motif_bb_coords,
					seg_motif_bb_coords
				)
				seg_designed_motif_bb_coords_aligned = (R.mm(seg_designed_motif_bb_coords.T)).T + t
				seg_motif_bb_rmsd = torch.sqrt(((seg_designed_motif_bb_coords_aligned - seg_motif_bb_coords)**2).sum(axis=1).mean())

				# Compute motif ca rmsd
				R, t = compute_rigid_alignment(
					seg_designed_motif_ca_coords,
					seg_motif_ca_coords
				)
				seg_designed_motif_ca_coords_aligned = (R.mm(seg_designed_motif_ca_coords.T)).T + t
				seg_motif_ca_rmsd = torch.sqrt(((seg_designed_motif_ca_coords_aligned - seg_motif_ca_coords)**2).sum(axis=1).mean())

				# Save
				motif_bb_rmsds.append(seg_motif_bb_rmsd)
				motif_ca_rmsds.append(seg_motif_ca_rmsd)

			# Aggregate
			motif_ca_rmsd = np.max(motif_ca_rmsds)
			motif_bb_rmsd = np.max(motif_bb_rmsds)

			# Save
			with open(motif_scores_filepath, 'a') as file:
				file.write('{},{:.3f},{:.3f}\n'.format(
					name, motif_ca_rmsd, motif_bb_rmsd
				))
