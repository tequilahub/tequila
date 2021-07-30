from setuptools import setup, find_packages
import os
import sys

# on Windows: remove jax and jaxlib from requirements.txt and add autgrad

def read_requirements(fname):
    with open('requirements.txt') as file:
        lines=file.readlines()
        requirements = [line.strip() for line in lines]
    return requirements

# get author and version information
VERSIONFILE="src/tequila/version.py"
info = {"__version__":None, "__author__":None}
with open(VERSIONFILE, "r") as f:
    for l in f.readlines():
        tmp = l.split("=")
        if tmp[0].strip().lower() in info:
            info[tmp[0].strip().lower()] = tmp[1].strip().strip("\"").strip("\'")

for k,v in info.items():
    if v is None:
        raise Exception("could not find {} string in {}".format(k,VERSIONFILE))

extras_3_6 = ['dataclasses']
extras_3_7 = []
additional = []

requirements = read_requirements('requirements.txt')

setup(
    name='tequila-basic',
    version=info["__version__"],
    author=info["__author__"],
    url="https://github.com/aspuru-guzik-group/tequila",
    description="Tequila is an abstract library for the development and prototyping of quantum algorithms.\nSee github for more information",
    author_email='jakob.kottmann@gmail.com',
    install_requires=requirements + additional,
    extras_require={
        ':python_version < "3.7"': extras_3_6,
        ':python_version == "3.7"': extras_3_7
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [os.path.join('src')]
    }
)
