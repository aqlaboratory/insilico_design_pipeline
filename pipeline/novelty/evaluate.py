import argparse

from pipeline.novelty.base import NoveltyPipeline


def main(args):

	# Pipeline
	pipeline = NoveltyPipeline(
		name=args.dataset,
		datadir=args.datadir
	)

	# Evaluate
	pipeline.evaluate(args.rootdir, args.num_cpus)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--rootdir', type=str, help='Root directory', required=True)
	parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
	parser.add_argument('--datadir', type=str, help='Dataset directory', required=True)
	parser.add_argument('--num_cpus', type=int, help='Number of CPUs', default=1)
	args = parser.parse_args()
	main(args)