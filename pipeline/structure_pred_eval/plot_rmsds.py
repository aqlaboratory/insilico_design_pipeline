import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str,
                        help='Directory containing RMSD scores')
    
    args = parser.parse_args()
    results_dir = args.results_dir
    graph_dir = results_dir + '/' + 'rmsd_graphs'

    if not os.path.isdir(graph_dir):
        os.mkdir(graph_dir)
    
    problems = [x for x in os.listdir(results_dir) if 'problem' in x]
    for problem in problems:
        name = problem.split('=')[1]
        rmsds = pd.read_csv(results_dir + '/' + problem + '/' + 'single_motif_scores.csv')['motif_ca_rmsd']
        plt.figure(figsize=(12, 8))
        plt.hist(rmsds)
        plt.title(f"Backbone RMSD for Full-Seq Conditioned Structures, {name} ({len(rmsds)} Samples)")
        plt.xlabel('RMSD')
        plt.ylabel('Counts')
        plt.savefig(graph_dir + '/' + name + '.png')
