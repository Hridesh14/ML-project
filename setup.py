from setuptools import find_packages,setup
from typing import List
hypen_e_dot ='-e .'

def get_requierments(file_path:str)->List[str]:
    '''
    This function will return the list of requirments
    '''
    requirements  = []
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements=[reg.replace('\n','') for reg in requirements]
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements
setup(
    name='ML_project',
    version='0.0.1',
    author='Hridesh Maithani',
    author_email='maithanihridesh9012@gmail.com',
    packages = find_packages(),
    install_requires= get_requierments('requirements.txt'))