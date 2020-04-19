import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='stablemodels',
    version='0.0.1',
    description="Learning stable deep dynamics models with controls",
    long_description=long_description,
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
