#!/usr/bin/env python

from distutils.core import setup

setup(
    name='PyTES',
    version='0.1',
    description='Python TES Utilities',
    author='Kazuhiro Sakai',
    author_email='sakai@astro.isas.jaxa.jp',
    url='https://github.com/robios/PyTES/',
    packages=['pytes'],
    package_dir={'pytes': 'src/pytes'},
)