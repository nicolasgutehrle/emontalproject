#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='GallicaParser',
      version='1.0',
      description='Framework to collect data from Gallica',
      url='#',
      author='Nicolas Gutehrlé',
      author_email='name@example.com',
    #   license='MIT',
      packages=find_packages(where='main', include=['*']),
      package_dir={"":"main"},
      zip_safe=False)