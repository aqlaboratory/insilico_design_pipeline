import os
import argparse
import numpy as np
import pandas as pd


def main(args):

	# Parse
	df = pd.read_csv(os.path.join(args.rootdir, 'info.csv'))
	df_designable = df[(df['scRMSD'] <= 2) & (df['pLDDT'] >= 70)]

	# Designability
	designability = len(df_designable) / len(df)

	# Tertiary diversity
	tertiary_diversity = len(df_designable['single_cluster_idx'].unique()) / len(df)

	# F1 score
	f1_score = 2 * designability * tertiary_diversity / (designability + tertiary_diversity)

	# PDB novelty
	pdb_novelty = None
	if 'max_pdb_tm' in df.columns:
	    pdb_novelty = len(df_designable[df_designable['max_pdb_tm'] < 0.5]['single_cluster_idx'].unique()) / len(df)

	# AFDB novelty
	afdb_novelty = None
	if 'max_afdb_tm' in df.columns:
	    afdb_novelty = len(df_designable[df_designable['max_afdb_tm'] < 0.5]['single_cluster_idx'].unique()) / len(df)

	# Print
	print('Designability:       {:.3f}'.format(designability))
	print('Tertiary diversity:  {:.3f}'.format(tertiary_diversity))
	print('F1 score:            {:.3f}'.format(f1_score))
	if pdb_novelty is not None:
		print('PDB novelty:         {:.3f}'.format(pdb_novelty))
	if afdb_novelty is not None:
		print('AFDB novelty:        {:.3f}'.format(afdb_novelty))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--rootdir', type=str, help='Root directory', required=True)
	args = parser.parse_args()
	main(args)