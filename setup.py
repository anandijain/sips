from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sips",
    version="0.14.1",
    description="tools for quantitative sports analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anand Jain",
    author_email="anandj@uchicago.edu",
    packages=["sips"],  # same as name
    url="https://github.com/anandijain/sips",
    install_requires=[
        "pandas",
        "requests",
        "beautifulsoup4",
        "numpy",
        "scikit-learn",
        "requests-futures",
        "google-cloud-profiler",
        "lxml",
        "flask",
    ],  # external packages as dependencies
    python_requires=">=3.6",
)
