import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    sys.stderr.write('Python >= 3.7 is required.')
    sys.exit(1)


setup(
    name='gym_pomdps',
    version='1.0.0',
    packages=find_packages(),
    package_data={'': ['*.pomdp']},
    test_suite='tests',
)
