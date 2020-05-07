from setuptools import setup, find_packages
import os
import sys

# on Windows: remove jax and jaxlib from requirements.txt and add autgrad

def read_requirements(fname):
    with open('requirements.txt') as file:
        lines=file.readlines()
        requirements = [line.strip() for line in lines]
    return requirements

extras_3_6 = ['dataclasses']
extras_3_7 = []
additional = []

requirements = read_requirements('requirements.txt')

setup(
    name='tequila',
    version="1.0",
    author='Jakob S. Kottmann, Sumner Alperin-Lea, Teresa Tamayo-Mendoza, Cyrille Lavigne, Alba Cervera-Lierta, Abhinav Anand, Maha Kesebi',
    author_email='jakob.kottmann@gmail.com',
    install_requires=requirements + additional,
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
