from setuptools import setup

with open('requirements.txt', "r") as file:
    requirements = file.readlines[r.strip() for r in requirements]

setup(
    name='OpenVQE',
    url='https://github.com/hsim13372/OpenVQE.git'
    authors='Hannah Sim, Jakob S. Kottmann, ...'
    packages=['openvqe']
    install_requires=['numpy']
    version='0.1'
    install_requires=requirements
    description=''
)
