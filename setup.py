from setuptools import setup

setup(
      name='pipeline',
      version='0.0.1',
      description='In-silico protein design evaluation pipeline',
      packages=['pipeline'],
      install_requires=[
            'tqdm',
            'numpy',
            'pandas'
      ],
)
