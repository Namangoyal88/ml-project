from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path) as file_object:
        
        requirements = file_object.readlines()
        requirements = [req.replace('\n', '').strip() for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    return requirements

setup(
    name='ml_projects',
    version='0.1.0',
    author='Naman Goyal',
    author_email='goyalnaman497@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A collection of machine learning projects'
)