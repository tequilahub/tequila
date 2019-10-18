from setuptools import setup, find_packages

requirements = None
with open('requirements.txt') as file:
    lines=file.readlines()
    requirements = [line.strip() for line in lines]

if requirements is None:
    raise Exception("Failed to read in requirements.txt")

setup(
    name='openvqe',
    version=__version__,
    author='Jakob S. Kottmann, Sumner Alperin-Lea, Abhinav Anand, Maha Kesebi',
    author_email='jakob.kottmann@gmail.com',
    install_requires=requirements,
    packages=find_packages(where='openvqe'),
    package_dir={'': 'openvqe'},
    include_package_data=True,
    package_data={
        '': [os.path.join('openvqe'),
             os.path.join('openvqe')]
    }
)
