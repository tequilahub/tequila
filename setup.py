from setuptools import setup, find_packages
import os
import sys

def read_requirements(fname):
    with open('requirements.txt') as file:
        lines=file.readlines()
        requirements = [line.strip() for line in lines]
    return requirements

extras_3_6 = ['dataclasses']
extras_3_7 = []
additional = []

setup(
    name='tequila',
    version="XXXX",
    author='Jakob S. Kottmann, Sumner Alperin-Lea, Teresa Tamayo-Mendoza, Cyrille Lavigne, Abhinav Anand, Maha Kesebi',
    author_email='jakob.kottmann@gmail.com',
    install_requires=[ # also requires jax+jaxlib or autograd
        'numpy',
        'scipy',
        'sympy',
        'setuptools',
        'pytest',
        'openfermion',
        'qulacs',
        'cirq',
        'sphinx',
        'm2r'       
    ] + additional,
    extras_require={
        ':python_version < "3.7"': extras_3_6,
        ':python_version == "3.7"': extras_3_7
    },
    packages=find_packages(include=['src','src/tequila',"src/tequila."]),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [os.path.join('src')]
    }
)
