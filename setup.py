from setuptools import setup

setup(
   name='sips',
   version='0.11',
   description='tools for RL sports betting',
   author='Anand Jain',
   author_email='anandj@uchicago.edu',
   packages=['sips'],  #same as name
   install_requires= \
           ['pandas', 'requests', 'beautifulsoup4', 'numpy', 'gym',
           'torch', 'scikit-learn', 'requests-futures',
           'matplotlib'] # external packages as dependencies
)
