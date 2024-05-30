import os
import glob
import argparse
import numpy as np
import pandas as pd


def main(args):

	# Initialize
	num_solved = 0
	total_num_unique_success = 0
	info = []

	# Iterate
	for dirname in glob.glob(os.path.join(args.rootdir, '*', '')):

		# Parse
		name = dirname.split('/')[-2].split('=')[-1]
		df = pd.read_csv(os.path.join(dirname, 'info.csv'))
		num_unique_success = len(df[
			(df['scRMSD'] <= 2) & (df['pLDDT'] >= 70) &
			(df['pAE'] <= 5) & (df['motif_bb_rmsd'] <= 1)
		]['single_cluster_idx'].unique())

		# Save
		if num_unique_success > 0:
			num_solved += 1
		total_num_unique_success += num_unique_success
		info.append((name, num_unique_success))

	# Print
	info = sorted(info, key=lambda x: x[1], reverse=True)
	print('Solved:                     {}'.format(num_solved))
	print('Number of unique successes: {}'.format(total_num_unique_success))
	for name, num_unique_success in info:
		print('\t{:<10}: {:>3}'.format(name, num_unique_success))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--rootdir', type=str, help='Root directory', required=True)
	args = parser.parse_args()
	main(args)