from setuptools import setup

setup(
    name="sips",
    version="0.15",
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
        "lxml",
        "flask"
    ],  # external packages as dependencies
)
