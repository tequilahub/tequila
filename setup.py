from setuptools import setup

requirements=''
with open('requirements.txt', "r") as file:
    requirements = file.readlines()
requirements = [r.strip() for r in requirements]

setup(
    name='OpenVQE',
    url='https://github.com/hsim13372/OpenVQE.git',
    author='Hannah Sim, Jakob S. Kottmann, ...',
    packages=['openvqe'],
    version='0.1',
    install_requires=requirements,
    description=''
)
