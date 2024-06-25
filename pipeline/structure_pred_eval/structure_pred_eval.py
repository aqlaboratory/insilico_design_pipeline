import argparse
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from pipeline.utils.align import compute_rigid_alignment

def compute_backbone_rmsd(motif_pdbs_dir, designs_dir, results_dir, verbose):
    """
    Compute backbone RMSDS. Outputs are stored in the 
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
        columns = ['domain', 'motif_ca_rmsd']
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
                        'ca_coords': []
                    }

                # Store coordinates
                coord = [
                    float(line[30:38].strip()),
                    float(line[38:46].strip()),
                    float(line[46:54].strip())
                ]
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
                            'ca_coords': []
                        }

                    # Store coordinates
                    coord = [
                        float(line[30:38].strip()),
                        float(line[38:46].strip()),
                        float(line[46:54].strip())
                    ]
                    if line[12:16].strip() == 'CA':
                        designed_motif_groups[group]['ca_coords'].append(coord)

        # Iterate
        assert len(motif_groups) == len(designed_motif_groups)
        motif_bb_rmsds, motif_ca_rmsds = [], []
        for group in motif_groups:

            # Parse
            seg_motif_ca_coords = motif_groups[group]['ca_coords']
            seg_designed_motif_ca_coords = designed_motif_groups[group]['ca_coords']
            assert len(seg_motif_ca_coords) == len(seg_designed_motif_ca_coords)

            # Convert to tensor
            seg_motif_ca_coords = torch.Tensor(seg_motif_ca_coords)
            seg_designed_motif_ca_coords = torch.Tensor(seg_designed_motif_ca_coords)

            # Compute motif ca rmsd
            R, t = compute_rigid_alignment(
                seg_designed_motif_ca_coords,
                seg_motif_ca_coords
            )
            seg_designed_motif_ca_coords_aligned = (R.mm(seg_designed_motif_ca_coords.T)).T + t
            seg_motif_ca_rmsd = torch.sqrt(((seg_designed_motif_ca_coords_aligned - seg_motif_ca_coords)**2).sum(axis=1).mean())

            motif_ca_rmsds.append(seg_motif_ca_rmsd)

        # Aggregate
        motif_ca_rmsd = np.max(motif_ca_rmsds)

        # Save
        with open(motif_scores_filepath, 'a') as file:
            file.write('{},{:.3f}\n'.format(
                name, motif_ca_rmsd#, motif_bb_rmsd
            ))

def evaluate(structure_dir, results_dir, verbose):
    for problem in os.listdir(structure_dir):
        results_subdir = results_dir + '/' + problem
        os.mkdir(results_subdir)
        problem_path = os.path.join(structure_dir, problem)
        if os.path.isdir(problem_path):
            motif_pdbs_path = os.path.join(problem_path, 'motif_pdbs')
            pdbs_path = os.path.join(problem_path, 'pdbs')
            if os.path.isdir(motif_pdbs_path) and os.path.isdir(pdbs_path):
                compute_backbone_rmsd(motif_pdbs_path, pdbs_path, results_subdir, verbose)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('structure_dir', type=str,
                        help='Directory containing problem directories with each problem consisting of a motif_pdbs and pdbs subdirectory')
    parser.add_argument('results_dir', type=str,
                        help='Directory where RMSD scores will be storied.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Whether to print detailed progress information.')
    
    args = parser.parse_args()
    evaluate(args.structure_dir, args.results_dir, args.verbose)
    


