import os
import glob
import argparse

from pipeline.diversity.base import DiversityPipeline


def main(args):

	# Define pipeline
	pipeline = DiversityPipeline()

	# Create directories
	if os.path.exists(os.path.join(args.rootdir, 'designs')):
		rootdirs = [args.rootdir]
	else:
		rootdirs = [
			'/'.join(subdir.split('/')[:-2])
			for subdir in glob.glob(os.path.join(args.rootdir, '*', 'designs', ''))
		]

	# Evaluate
	for rootdir in rootdirs:
		pipeline.evaluate(rootdir, args.num_cpus)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--rootdir', type=str, help='Root directory', required=True)
	parser.add_argument('--num_cpus', type=int, help='Number of CPUs', default=1)
	args = parser.parse_args()
	main(args)