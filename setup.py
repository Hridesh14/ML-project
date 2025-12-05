from setuptools import find_packages,setup
from typing import List
hyypen_e_dot ='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirments

    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace('\n','')for req in requirements]

        if hyypen_e_dot in requirements:
            requirements.remove(hyypen_e_dot)


setup(
    name='ML_project',
    version='0.0.1',
    author='Hridesh Maithani',
    author_email='maithanihridesh9012@gmail.com',
    packages=get_requirements('requirements.txt')
)      