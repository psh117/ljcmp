## setup.py
from setuptools import find_packages, setup

setup(
    name='ljcmp',
    version='0.1',
    description='Constrained Motion Planning with Learned Latent Spaces',
    packages=find_packages(include=['ljcmp', 'ljcmp.*']),
    install_requires=[
        'pytorch_lightning',
        'pyquaternion',
        'torch',
        'matplotlib',
        'scipy',
        'anytree',
        'numpy'
    ]
)