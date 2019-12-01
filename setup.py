from setuptools import setup

setup(
    name="sips",
    version="0.14",
    description="tools for quantitative sports betting",
    author="Anand Jain",
    author_email="anandj@uchicago.edu",
    packages=["sips"],  # same as name
    install_requires=[
        "pandas",
        "requests",
        "beautifulsoup4",
        "numpy",
        "scikit-learn",
        "requests-futures",
        "google-cloud-profiler",
        "lxml"
    ],  # external packages as dependencies
)
