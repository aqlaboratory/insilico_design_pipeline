import os
import glob
import gzip
import shutil
import subprocess
import numpy as np
import pandas as pd

from pipeline.utils.process import run_parallel


class NoveltyPipeline():
	"""
	Novelty evaluation pipeline.

	For each designed structure, this pipeline computes its TM score 
	to all structures in the reference dataset, finds the closest 
	structure in the reference dataset (maximum TM score) and stores  
	this statistic by updating the file named 'info.csv' in the 
	root directory. We assume that the standard pipeline is executed 
	before this.
	"""

	def __init__(
		self,
		name,
		datadir,
		tm_align_exec='packages/TMscore/TMalign'
	):
		"""
		Args:
			name:
				Reference dataset name used to ditinguish between evaluation 
				outputs from different reference datasets. It is used in defining 
				column names for the final output table on novelty statistics.
			datadir:
				Directory for the reference dataset.
			tm_align_exec:
				Path to TMalign executable. Default to 'packages/TMalign/TMalign'.
		"""
		self.name = name.lower()
		self.datadir = datadir
		self.tm_align_exec = tm_align_exec

	def evaluate(self, rootdir, num_processes):
		"""
		Evaluate a set of designed structures on novelty. Outputs are stored in 
		the statistics file named 'info.csv' by concatenating novelty statistics 
		into the original file.

		Args:
			rootdir:
				Root directory consist of
					-	a subdirectory named 'pdbs', where each file contains a 
						generated structure in the PDB format
					-	[Optional] a subdirectory named 'motif_pdbs', where each 
						corresponding file (same filename as the filename in the 
						'pdbs' subdirectory) contains the motif structure, aligned 
						in residue indices with the generated structure and stored 
						in the PDB format
					- 	a subdirectory named 'designs', where each file is the most 
						similar structure (predicted by the folding model) to the 
						generated structure and is stored in a PDB format
					-	a file named 'info.csv', which contains aggregated evaluation 
						statistics for the set of generated structures.
			num_processes:
				Number of processes/CPUs used for running novelty evaluation.
		"""
		
		#################
		###   Setup   ###
		#################

		# Temporary directories that are cleaned at the end of the process
		self.tempdirs = []

		# Check for input directory
		assert os.path.exists(rootdir), 'Missing root directory'
		designs_dir = os.path.join(rootdir, 'designs')
		assert os.path.exists(designs_dir), 'Missing designs directory'

		# Create output directory
		novelties_dir = os.path.join(rootdir, 'novelties')
		assert not os.path.exists(novelties_dir), 'Output novelties directory existed'
		os.mkdir(novelties_dir)
		self.tempdirs.append(novelties_dir)

		# Set up reference dataset
		reference_pdbs = self._get_reference_pdbs(rootdir, num_processes)
		design_pdbs = glob.glob(os.path.join(designs_dir, '*.pdb'))
		print(f'Number of designs:    {len(design_pdbs)}')
		print(f'Number of references: {len(reference_pdbs)}')

		##################
		###   Define   ###
		##################

		def process(i, tasks, params):

			# Set up output file
			novelties_filepath = os.path.join(params['output_dir'], f'{i}.csv')
			with open(novelties_filepath, 'w') as file:
				columns = ['design', 'reference', 'tm']
				file.write(','.join(columns) + '\n')

			# Iterate by all designs
			for design_filepath in tasks:

				# Define
				design_name = design_filepath.split('/')[-1].split('.')[0]
				reference_names, reference_tm_scores = [], []

				# Iterate by all references
				for reference_filepath in params['reference_pdbs']:

					# Execute
					output_filepath = os.path.join(params['output_dir'], f'process_{i}.temp.txt')
					cmd = '{} {} {} -fast > {}'.format(
						params['tm_align_exec'],
						design_filepath,
						reference_filepath,
						output_filepath
					)
					subprocess.call(cmd, shell=True)

					# Parse
					with open(output_filepath) as file:
						for line in file:
							if line.startswith('TM-score') and 'Chain_1' in line:
								reference_names.append(reference_filepath.split('/')[-1].split('.')[0])
								reference_tm_scores.append(float(line.split('(')[0].split('=')[-1].strip()))
					os.remove(output_filepath)

				# Aggregate
				closest_reference_idx = np.argmax(reference_tm_scores)
				closest_reference_name = reference_names[closest_reference_idx]
				closest_reference_tm_score = reference_tm_scores[closest_reference_idx]

				# Save
				with open(novelties_filepath, 'a') as file:
					file.write('{},{},{:.3f}\n'.format(
						design_name,
						closest_reference_name,
						closest_reference_tm_score
					))

		###################
		###   Process   ###
		###################

		# Run
		run_parallel(
			num_processes=num_processes,
			fn=process,
			tasks=design_pdbs,
			params={
				'tm_align_exec': self.tm_align_exec,
				'reference_pdbs': reference_pdbs,
				'output_dir': novelties_dir
			}
		)

		# Aggregate
		self._aggregate(novelties_dir, rootdir)

		#################
		###   Clean   ###
		#################

		for tempdir in self.tempdirs:
			shutil.rmtree(tempdir)

	def _get_reference_pdbs(self, rootdir, num_processes):
		"""
		Set up reference datasets for evaluation.
		"""

		def process(i, tasks, params):
			for filepath in tasks:
				name = filepath.split('/')[-1].split('.')[0]
				output_filepath = os.path.join(params['output_dir'], f'{name}.pdb')
				with gzip.open(filepath, 'rb') as f_in:
					with open(output_filepath, 'wb') as f_out:
						shutil.copyfileobj(f_in, f_out)

		# Set up temporary directory
		references_dir = os.path.join(rootdir, 'references')
		assert not os.path.exists(references_dir), 'Output references directory existed'
		os.mkdir(references_dir)
		self.tempdirs.append(references_dir)

		# Process references
		input_filepaths = glob.glob(os.path.join(self.datadir, '*.pdb.gz'))
		run_parallel(
			num_processes=num_processes,
			fn=process,
			tasks=input_filepaths,
			params={
				'output_dir': references_dir
			}
		)

		return glob.glob(os.path.join(references_dir, '*.pdb'))

	def _aggregate(self, novelties_dir, output_dir):
		"""
		Aggregate information and update statistic file.
		"""

		# Create output filepath
		assert os.path.exists(novelties_dir), 'Missing output novelties directory'
		info_filepath = os.path.join(output_dir, 'info.csv')
		assert os.path.exists(info_filepath), 'Missing output info filepath'

		# Process
		df_novelties = pd.concat([
			pd.read_csv(filepath)
			for filepath in glob.glob(os.path.join(novelties_dir, '*.csv'))
		]).rename(
			columns={
				'design': 'domain',
				'reference': f'max_{self.name}_name',
				'tm': f'max_{self.name}_tm'
			}
		)

		# Merge
		df = pd.read_csv(info_filepath)
		df = df.merge(df_novelties, on='domain')

		# Save
		df.to_csv(info_filepath, index=False)
