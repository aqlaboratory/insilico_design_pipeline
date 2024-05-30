import os
import glob
import argparse
from tqdm import tqdm
from pipeline.utils.process import MultiProcessor


def load_inverse_fold_model(name, device):
	"""
	Load inverse folding model.

	Args:
		name:
			Name of inverse folding model. Currently support: proteinmpnn.
		device:
			Device name (for example, cuda:0).

	Returns:
		An inverse folding model (an instance of a derived class whose base 
		class is defined in pipeline/models/inverse_folds/base.py).
	"""
	print('Loading inverse fold model')
	if name == 'proteinmpnn':
		from pipeline.models.inverse_folds.proteinmpnn import ProteinMPNN
		return ProteinMPNN(device=device)
	else:
		print('Invalid inverse fold model: {}'.format(name))
		exit(0)

def load_fold_model(name, device):
	"""
	Load folding (structure prediction) model.

	Args:
		name:
			Name of folding (structure prediction) model. Currently support:
			esmfold.
		device:
			Device name (for example, cuda:0).

	Returns:
		A structure prediction model (an instance of a derived class whose
		base class is defined in pipeline/models/folds/base.py).
	"""
	print('Loading fold model')
	if name == 'esmfold':
		from pipeline.models.folds.esmfold import ESMFold
		return ESMFold(device=device)
	else:
		print('Invalid fold model: {}'.format(name))
		exit(0)

def load_pipeline(name, inverse_fold_model, fold_model):
	"""
	Load standard evaluation pipeline.

	Args:
		name:
			Name of standard evaluation pipeline. Currently support:
			unconditional, scaffold.
		inverse_fold_model:
			An inverse folding model (an instance of a derived class whose base 
			class is defined in pipeline/models/inverse_folds/base.py).
		fold_model:
			A structure prediction model (an instance of a derived class whose
			base class is defined in pipeline/models/folds/base.py).

	Returns:
		An standard evaluation pipeline (an instance of a derived class whose
		base class is defined in pipeline/standard/base.py).
	"""
	print('Load pipeline')
	if name == 'unconditional':
		from pipeline.standard.unconditional import UnconditionalPipeline
		return UnconditionalPipeline(inverse_fold_model, fold_model)
	elif name == 'scaffold':
		from pipeline.standard.scaffold import ScaffoldPipeline
		return ScaffoldPipeline(inverse_fold_model, fold_model)
	else:
		print('Invalid pipeline: {}'.format(name))
		exit(0)


class EvaluationRunner(MultiProcessor):
	"""
	A multi-processing runner for standard evaluation, whose base class is
	defined in pipeline/utils/process.py.
	"""

	def create_tasks(self, params):
		"""
		Define a set of tasks to be distributed across processes.

		Args:
			params:
				A dictionary of parameters.

		Returns:
			tasks:
				A list of tasks to be distributed across processes, where 
				each task is represented as a dictionary of task-specific 
				parameters.
		"""

		# Load directories
		if os.path.exists(os.path.join(params['rootdir'], 'pdbs')):
			rootdirs = [params['rootdir']]
		else:
			rootdirs = [
				'/'.join(subdir.split('/')[:-2])
				for subdir in glob.glob(os.path.join(params['rootdir'], '*', 'pdbs', ''))
			]

		# Create tasks
		tasks = [
			{ 'rootdir': rootdir }
			for rootdir in rootdirs
		]

		return tasks

	def create_constants(self, params):
		"""
		Define a dictionary of constants shared across processes.

		Args:
			params:
				A dictionary of parameters.

		Returns:
			constants:
				A dictionary of constants shared across processes.
		"""
		
		# Define
		names = [
			'version',
			'verbose',
			'inverse_fold_model_name',
			'fold_model_name'
		]

		# Create constants
		constants = dict([(name, params[name]) for name in names])

		return constants

	def execute(self, constants, tasks, device):
		"""
		Execute a set of assigned tasks on a given device.

		Args:
			constants:
				A dictionary of constants.
			tasks:
				A list of tasks, where each task is represented as a 
				dictionary of task-specific parameters.
			device:
				Name of device to execute on.
		"""

		# Create pipeline
		pipeline = load_pipeline(
			constants['version'],
			load_inverse_fold_model(constants['inverse_fold_model_name'], device),
			load_fold_model(constants['fold_model_name'], device)
		)

		# Evaluate
		for task in tqdm(tasks, desc=device):
			pipeline.evaluate(task['rootdir'], verbose=constants['verbose'])


def main(args):
	
	# Define multiprocessor runner
	runner = EvaluationRunner()

	# Run
	runner.run(vars(args), args.num_processes, args.num_devices)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--rootdir', type=str, help='Root directory', required=True)
	parser.add_argument('--version', type=str, help='Pipeline version', required=True)
	parser.add_argument('--verbose', help='Verbose', action='store_true', default=False)
	parser.add_argument('--inverse_fold_model_name', type=str, help='Inverse fold model name', default='proteinmpnn')
	parser.add_argument('--fold_model_name', type=str, help='Fold model name', default='esmfold')
	parser.add_argument('--num_processes', type=int, help='Number of processes', default=1)
	parser.add_argument('--num_devices', type=int, help='Number of GPU devices', default=1)
	args = parser.parse_args()
	main(args)
