import os
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd

from pipeline.utils.cluster import hcluster
from pipeline.utils.process import run_parallel


class DiversityPipeline():
	"""
	Diversity evaluation pipeline. 

	This pipeline first computes the pairwise TM score among the set of designed 
	structures, where each structure is predicted by the structure prediction model 
	and is most similar to its corresponding generated structure. It then performs 
	hierarchical clustering on this set of designed structures and clusters them 
	based on structural similarity as measured by TM score. We assume that the 
	standard pipeline is executed before this.
	"""

	def __init__(
		self,
		postfix='',
		max_ctm_threshold=0.6,
		tm_align_exec='packages/TMscore/TMalign'
	):
		"""
		Args:
			postfix:
				Additional postfix defined to distinguish output from different 
				diversity pipeline. It is used in defining column names for the
				final output table on diversity statistics. Default to an empty 
				string.
			max_ctm_threshold:
				Maximum TM score threshold between clusters. Default to 0.6.
			tm_align_exec:
				Path to TMalign executable. Default to 'packages/TMalign/TMalign'.
		"""
		self.postfix = postfix
		self.tm_align_exec = tm_align_exec
		self.max_ctm_threshold = max_ctm_threshold

	def evaluate(self, rootdir, num_processes):
		"""
		Evaluate a set of designed structures on diversity. Outputs are stored in 
		the statistics file named 'info.csv' by concatenating diversity statistics 
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
				Number of processes/CPUs used for running diversity evaluation.
		"""

		# Temporary directories that are cleaned at the end of the process
		self.tempdirs = []

		# Check for input files/directories
		info_filepath = os.path.join(rootdir, 'info.csv')
		assert os.path.exists(info_filepath), 'Missing input info filepath'
		designs_dir = os.path.join(rootdir, 'designs')
		assert os.path.exists(designs_dir), 'Missing input designs directory'

		# Check for existing clustering information
		df = pd.read_csv(info_filepath)
		assert f'single_cluster_idx{self.postfix}' not in df.columns, 'Single cluster information existed'
		assert f'complete_cluster_idx{self.postfix}' not in df.columns, 'Complete cluster information existed'
		assert f'average_cluster_idx{self.postfix}' not in df.columns, 'Average cluster information existed'

		# Process
		scores_dir = self._compute_scores(designs_dir, rootdir, num_processes)
		self._compute_clusters(scores_dir, rootdir)

		# Clean
		for tempdir in self.tempdirs:
			shutil.rmtree(tempdir)

	def _compute_scores(self, designs_dir, output_dir, num_processes):
		"""
		Compute pairwise TM score among the set of designed structures.

		Args:
			designs_dir:
				A directory of designed structure, where each file is the 
				most similar structure (predicted by the folding model) to 
				the generated structure and is stored in a PDB format.
			output_dir:
				Base output directory.
			num_processes:
				Number of processes/CPUs used for running diversity evaluation.

		Returns:
			scores_dir:
				Output directory (specified as [output_dir]/scores), where each 
				file stores the processed output from each process/CPU and each 
				line in the file stores the TM score between a pair of designed 
				structures (in the format of 'name1,name2,tmscore').
		"""

		#################
		###   Setup   ###
		#################

		# Create output directory
		scores_dir = os.path.join(output_dir, 'scores')
		assert not os.path.exists(scores_dir), 'Output scores directory existed'
		os.mkdir(scores_dir)
		self.tempdirs.append(scores_dir)

		# Create tasks
		tasks = []
		filepaths = glob.glob(os.path.join(designs_dir, '*.pdb'))
		for idx1, filepath1 in enumerate(filepaths):
			for idx2, filepath2 in enumerate(filepaths):
				if idx1 < idx2:
					tasks.append((filepath1, filepath2))

		##################
		###   Define   ###
		##################

		def process(i, tasks, params):

			# Set up output file
			scores_filepath = os.path.join(params['output_dir'], f'{i}.csv')
			with open(scores_filepath, 'w') as file:
				columns = ['domain_1', 'domain_2', 'tm']
				file.write(','.join(columns) + '\n')

			# Iterate
			for (design_filepath_1, design_filepath_2) in tasks:

				# Parse filepath
				domain_1 = design_filepath_1.split('/')[-1].split('.')[0]
				domain_2 = design_filepath_2.split('/')[-1].split('.')[0]

				# Compare pdb files
				output_filepath = os.path.join(params['output_dir'], f'output_{i}.txt')
				subprocess.call(f'{self.tm_align_exec} {design_filepath_1} {design_filepath_2} -fast > {output_filepath}', shell=True)
				
				# Parse TMalign output 
				rows = []
				with open(output_filepath) as file:
					for line in file:
						if line.startswith('TM-score') and 'Chain_1' in line:
							tm = float(line.split('(')[0].split('=')[-1].strip())
							rows.append((domain_1, domain_2, tm))
						if line.startswith('TM-score') and 'Chain_2' in line:
							tm = float(line.split('(')[0].split('=')[-1].strip())
							rows.append((domain_2, domain_1, tm))

				# Clean up
				os.remove(output_filepath)

				# Save
				with open(scores_filepath, 'a') as file:
					for domain_1, domain_2, tm in rows:
						file.write('{},{},{:.3f}\n'.format(domain_1, domain_2, tm))

		###################
		###   Process   ###
		###################

		run_parallel(
			num_processes=num_processes,
			fn=process,
			tasks=tasks,
			params={
				'tm_align_exec': self.tm_align_exec,
				'output_dir': scores_dir
			}
		)

		return scores_dir

	def _compute_clusters(self, scores_dir, output_dir):
		"""
		Perform hierarchical clustering on the set of designed structures, 
		based on precomputed pairwise TM scores. Outputs are stored in 
		the file named 'info.csv' under the root directory, by concatenating
		clustering statistics into the original file.

		Args:
			scores_dir:
				A directory where each file stores the processed output from 
				each process/CPU and each line in the file stores the TM score 
				between a pair of designed structures (in the format of 
				'name1,name2,tmscore').
			output_dir:
				Base output directory.
		"""

		# Create output filepath
		assert os.path.exists(scores_dir), 'Missing input scores directory'
		info_filepath = os.path.join(output_dir, 'info.csv')
		assert os.path.exists(info_filepath), 'Missing input info filepath'
		clusters_filepath = os.path.join(output_dir, f'single_clusters.csv')
		assert not os.path.exists(clusters_filepath), 'Output clusters filepath existed'
		with open(clusters_filepath, 'w') as file:
			columns = [
				'domain',
				f'single_cluster_idx{self.postfix}',
				f'complete_cluster_idx{self.postfix}',
				f'average_cluster_idx{self.postfix}'
			]
			file.write(','.join(columns) + '\n')

		# Create index map
		domains, domain_idx_map = [], {}
		df = pd.read_csv(info_filepath)
		for (idx, row) in df.iterrows():
			domain = row['domain']
			domains.append(domain)
			domain_idx_map[domain] = len(domain_idx_map)

		# Load scores
		df_scores = pd.concat([
			pd.read_csv(filepath)
			for filepath in glob.glob(os.path.join(scores_dir, '*.csv'))
		])

		# Create distance matrix
		dists = np.zeros((len(domains), len(domains)))
		for (idx, row) in df_scores.iterrows():
			domain_idx_1 = domain_idx_map[row['domain_1']]
			domain_idx_2 = domain_idx_map[row['domain_2']]
			dists[domain_idx_1][domain_idx_2] = row['tm']

		# Compute clusters
		columns = []
		linkages = ['single', 'complete', 'average']
		for linkage in linkages:

			# Perform hierarchical clustering
			clusters = hcluster(dists, linkage, max_ctm_threshold=self.max_ctm_threshold)

			# Map domain to cluster idx
			domain_cluster_idx_map = {}
			for cluster_idx, cluster in enumerate(clusters):
				for domain_idx in cluster:
					domain = domains[domain_idx]
					domain_cluster_idx_map[domain] = cluster_idx

			# Create column
			columns.append([domain_cluster_idx_map[domain] for domain in domains])

		# Save cluster information
		with open(clusters_filepath, 'a') as file:
			for i, domain in enumerate(domains):
				file.write('{},{},{},{}\n'.format(domain, columns[0][i], columns[1][i], columns[2][i]))

		# Merge
		df_clusters = pd.read_csv(clusters_filepath)
		df = df.merge(df_clusters, on='domain')

		# Save
		df.to_csv(info_filepath, index=False)

		# Clean
		os.remove(clusters_filepath)
