# In-silio Protein Design Pipeline

This repository contains the in-silico protein design and evaluation pipeline that we used for assessing [Genie 2](https://arxiv.org/abs/2405.15489). We set this up separately from [Genie 2 repository](https://github.com/aqlaboratory/genie2) to facilitate assessments of different structure-based protein diffusion models. The pipeline consists of:
- assessment on designability through self-consistency pipeline
- assessment on secondary diversity through P-SEA algorithm
- assessment on tertiary diversity through hierarchical clustering
- assessment on novelty

## Set up
Assume the environment has a cuda-compatiable PyTorch installed and Python <= 3.9. For example, on our own machine, the environment is created and initialized by running.

```
python3.9 -m venv insilico_pipeline_venv
source insilico_pipeline_venv/bin/activate
module load cuda11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

The setup process consists of three parts:

- Set up the pipeline package and additional packages (TMscore and TMalign) by running

  ```
  bash scripts/setup/setup.sh
  ```

- Set up an inverse folding model and its dependencies by running

  ```
  bash scripts/setup/inverse_folds/setup_[INVERSE_FOLD_MODEL_NAME].sh
  ```

  Our current pipeline supports ProteinMPNN (`proteinmpnn`) and we intend to extend this to include more inverse folding models.

- Set up a folding model and its dependencies by running

  ```
  bash scripts/setup/folds/setup_[FOLD_MODEL_NAME].sh
  ```
  
  Our current pipeline supports ESMFold (`esmfold`) and we intend to extend this to include more folding models.

### Additional notes

When setting up the environment for ESMFold, we install OpenFold v1.0.1 to ensure compatibility. One known [issue](https://github.com/aqlaboratory/openfold/issues/276) for this OpenFold installation is its compatibility with deepspeed. This raises `AttributeError: module 'deepspeed.utils' has no attribute 'is_initialized'` when running the pipeline and could be fixed by replacing all occurences of `deepspeed.utils.is_initialized()` with `deepspeed.comm.comm.is_initialized()`.

## Pipelines

Our design package consists of three separate pipelines:
- standard pipeline for assessing designability and secondary diversity
- diversity pipeline for assessing tertiary diversity
- novelty pipeline for assessing novelty with respect to a reference dataset.

### Standard pipeline (`pipeline/standard`)

Evaluate a set of generated structures by running

```
python pipeline/standard/evaluate.py --version [VERSION] --rootdir [ROOTDIR]
```

Our standard pipeline currently supports evaluation of structures from unconditional generation (by setting version to `unconditional`) and motif scaffolding (`scaffold`). For both modes, we assume that the root directory contains a folder named `pdbs`, which contains the PDB files of generated structures to be evaluated. For motif scaffolding, we additionally assume that the root directory contains a folder named `motif_pdbs`, which contains the PDB files of the corresponding motif structures (with the same filename as the generated structure and residue index aligned). Note that for motif scaffolding, we also support evaluations of multiple problems at the same time. This means that the root directory could contain a list of subdirectories, each of which consists of a `pdbs` and `motif_pdbs` folder detailed above. When evaluating multiple motif scaffolding problems, our pipeline supports distribution of tasks across multiple GPUS by adding the following flags `--num_devices [NUM_GPUS] --num_processes [NUM_GPUS]`.

Evaluation results are stored in the root directory, which contains:
- a directory named `designs`, where each PDB file stores the fold model predicted structure that is most similar to the corresponding generated structure;
- a csv file named `info.csv`, which contains evaluation statistics for the set of generated structures. Information on columns is summarized in the table below.

  | Column | Description |
  | :--- | :--------------------------- |
  | `domain`                   | Name of generated structure |
  | `seqlen`                   | Sequence length of generated structure |
  | `scRMSD`                   | RMSD between the generated structure and the most <br>similar structure predicted by the specified fold model |
  | `scTM`                     | TM score between the generated structure and the most <br>similar structure predicted by the specified fold model |
  | `pLDDT`                    | Local confidence from the specified fold model, <br>averaged across all residues |
  | `pAE`                      | Confidence from the specified fold model in the <br>relative position of two residues, averaged across all <br>residue-residue pairs |
  | `generated_pct_helix`      | Percentage of helices in the generated structure |
  | `generated_pct_strand`     | Percentage of strands in the generated structure |
  | `generated_pct_ss`         | Percentage of helices and strands in the <br>generated structure |
  | `generated_pct_left_helix` | Percentage of left-handed helices in the <br>generated structure |
  | `designed_pct_helix`       | Percentage of helices in the most similar structure <br>predicted by the specific fold model |
  | `designed_pct_strand`      | Percentage of strands in the most similar structure <br>predicted by the specific fold model |
  | `designed_pct_ss`          | Percentage of helices and strands in the most similar <br>structure predicted by the specific fold model |
  | `designed_pct_left_helix`  | Percentage of left-handed helices in the most similar <br>structure predicted by the specific fold model |

Note that for secondary structure evaluations, we use the P-SEA algorithm, which allows us to predict secondary structures based on Ca atoms only.

### Diversity pipeline (`pipeline/diversity`)

Assume that a set of generated structure is assessed by the above standard pipeline. Evaluate this set of generated structures on tertiary diversity by running

```
python pipeline/diversity/evaluate.py --rootdir [ROOTDIR] --num_cpus [NUM_CPUS]
```

Results are stored by updating `info.csv` in the root directory to include

| Column | Description |
| :------- | :--------- |
| `single_cluster_idx`   | Index of cluster that the generated structure belongs <br>(hierarchically clustered via single linkage) |
| `complete_cluster_idx` | Index of cluster that the generated structure belongs <br>(hierarchically clustered via complete linkage) |
| `average_cluster_idx`  | Index of cluster that the generated structure belongs <br>(hierarchically clustered via average linkage) |

Note that for hierarchical clustering, we use TMalign to compute pairwise TM scores among all generated structures and a TM score threshold of 0.6 in the clustering process.

### Novelty pipeline (`pipeline/novelty`)

Assume that a set of generated structure is assessed by the above standard pipeline. Evaluate this set of generated structures on novelty by running

```
python pipeline/novelty/evaluate.py --rootdir [ROOTDIR] --dataset [DATASET] --datadir [DATADIR] --num_cpus [NUM_CPUS]
```

where `DATASET` is the name of the reference dataset and `DATADIR` is the directory for the reference dataset (with each reference structure stored in a PDB format). Results are stored by updating `info.csv` in the root directory to include

| Column | Description |
| :------- | :--------- |
| `max_[DATASET]_name` | Name of structure in the dataset that is most <br>similar to the generated structure |
| `max_[DATASET]_tm`   | TM score between the generated structure and the <br>most similar structure in the dataset |

## Examples

In the `examples` directory, we provide three examples (together with their correponding outputs) to demonstrate the input and output to our evaluation pipeline. Examples include:
-	`unconditional`: evaluation of a unconditionally generated structure
-	`scaffold_single`: evaluation of a conditionally generated structure, whose generation is conditioned on a single functional motif
-	`scaffold_multi`: evaluation of a conditionally generated structure, whose generation is conditioned on multiple functional motifs

## Profiling

### Unconditional generation

Assume that the standard (designability) and diversity pipelines are run. To show the evaluation metrics on the set of generated structures, run

```
python scripts/analysis/profile_unconditional.py --rootdir [ROOTDIR]
```

This reports designability, diversity and F1 score on the set of generated structures. It also reports PDB novelty and/or AFDB novelty, provided that the corresponding novelty pipeline is run. Details on these evaluation metrics are found in the [Genie 2](https://arxiv.org/abs/2405.15489) paper.

### Motif scaffolding

Assume that the standard (designability) and diversity pipelines are run. To show the evaluation metrics on the set of generated structures, run

```
python scripts/analysis/profile_scaffold.py --rootdir [ROOTDIR]
```

This reports the number of solved motif scaffolding problems and the total number of unique clusters, aggregated across all problems. Details on these evaluation metrics are found in the [Genie 2](https://arxiv.org/abs/2405.15489) paper. Here, we assume that the root directory contains a set of subdirectories, where each subdirectory starts with a prefix of `motif=` and contains inputs and outputs for a motif scaffolding problem (check out `examples/scaffold_single` and `examples/scaffold_multi` for detailed examples).

