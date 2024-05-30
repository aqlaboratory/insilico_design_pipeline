import os
import math
from multiprocessing import Process
from abc import ABC, abstractmethod


def run_parallel(num_processes, fn, tasks, params):
	"""
	Run parallel with multiple CPUs.

	Args:
		num_processes:
			Number of processes/CPUs.
		fn:
			Execution function.
		tasks:
			A list of tasks, where each task is defined as a dictionary 
			of task-specific parameters.
		params:
			A dictionary of constants shared across processes.
	"""

	# Start parallel processes
	processes = []
	binsize = math.ceil(len(tasks) / num_processes)
	for i in range(num_processes):
		p = Process(
			target=fn,
			args=(
				i,
				tasks[binsize*i:binsize*(i+1)],
				params
			)
		)
		p.start()
		processes.append(p)

	# Wait for completion
	for p in processes:
		p.join()


class MultiProcessor(ABC):
	"""
	Base class for multiprocessing.
	"""

	@abstractmethod
	def create_tasks(self, params):
		"""
		Define a list of tasks to be distributed across processes, where each 
		task is defiend as a dictionary of task-specific parameters.

		Args:
			params:
				A dictionary of parameters.

		Returns:
			A list of tasks, where each task is defiend as a dictionary of 
			task-specific parameters.
		"""
		raise NotImplemented

	@abstractmethod
	def create_constants(self, params):
		"""
		Define a dictionary of constants shared across processes.

		Args:
			params:
				A dictionary of parameters.

		Returns:
			A dictionary of constants shared across processes.
		"""
		raise NotImplemented

	@abstractmethod
	def execute(self, constants, tasks, device):
		"""
		Execute a list of tasks on the given device.

		Args:
			constants:
				A dictionary of constants.
			tasks:
				A list of tasks, where each task is defiend as a dictionary 
				of task-specific parameters.
			device:
				Device to run on.
		"""
		raise NotImplemented

	def run(self, params, num_processes, num_devices):
		"""
		Run in parallel based on input parameters/configurations.

		Args:
			params:
				A dictionary of parameters/configurations.
			num_processes:
				Number of processes to execute tasks.
			num_devices:
				Number of GPUs availble.
		"""

		# Create tasks
		tasks = self.create_tasks(params)

		# Create constants
		constants = self.create_constants(params)

		# Start parallel processes
		processes = []
		binsize = math.ceil(len(tasks) / num_processes)
		for i in range(num_processes):
			device = f'cuda:{i % num_devices}' if num_devices > 0 else 'cpu'
			p = Process(
				target=self.execute,
				args=(
					constants,
					tasks[binsize*i:binsize*(i+1)],
					device
				)
			)
			p.start()
			processes.append(p)

		# Wait for completion
		for p in processes:
			p.join()
