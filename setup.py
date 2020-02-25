from setuptools import setup, find_packages
import os

requirements = None
with open('requirements.txt') as file:
    lines=file.readlines()
    requirements = [line.strip() for line in lines]

if requirements is None:
    raise Exception("Failed to read in requirements.txt")

setup(
    name='tequila',
    version="XXXX",
    author='Jakob S. Kottmann, Sumner Alperin-Lea, Teresa Tamayo-Mendoza, Cyrille Lavigne, Abhinav Anand, Maha Kesebi',
    author_email='jakob.kottmann@gmail.com',
    install_requires=[
        'numpy',
        'scipy',
        'sympy',
        'jax',
        'jaxlib',
        'setuptools',
        'pytest',
        'openfermion',
        'qulacs',
        'cirq',
        'sphinx',
        'm2r'       
    ],
    extras_require={
        ':python_version < "3.7"': [
            'dataclasses',
        ],
    },
    packages=find_packages(include=['src','src/tequila',"src/tequila."]),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [os.path.join('src')]
    }
)
