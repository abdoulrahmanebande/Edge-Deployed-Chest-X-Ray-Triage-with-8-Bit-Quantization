from setuptools import find_packages, setup 
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str] :
  requirements = [] 
  
  with open (file_path) as file_object:
    requirements = file_object.readlines()
    requirements = [req.replace('\n', '').strip() for req in requirements]
    
    if HYPHEN_E_DOT in requirements:
      requirements.remove(HYPHEN_E_DOT)
      
  return requirements

setup (
  name='Edge-Deployed Chest X-Ray Triage',
  description='Lightweight chest X-ray pneumonia triage web app using 8-bit quantized (INT8) MobileNetV3-Small and Grad-CAM visual explainability for fast CPU inference in low-resource clinics.',
  version='0.0.1',
  author='Abdoul-Rahmane BANDE',
  author_email='abdoulrahmanebande@gmail.com',
  packages= find_packages(),
  install_requires= get_requirements('requirements.txt')
)