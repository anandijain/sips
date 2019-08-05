from setuptools import setup

setup(
   name='sips',
   version='0.1',
   description='tools for RL sports betting',
   author='Anand Jain',
   author_email='anandj@uchicago.edu',
   packages=['sips'],  #same as name
   install_requires= \
           ['pandas', 'requests', 'beautifulsoup4', 'numpy', 'gym',
           'torch', 'seaborn', 'scikit-learn',
           'matplotlib'] # external packages as dependencies
)