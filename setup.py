#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='th_fallfilm',
    # la version du code
    version="0.1",
    # Liste les packages à insérer dans la distribution
    packages=find_packages(),

    author="Nicolas CELLIER",
    author_email="contact@nicolas-cellier.net",
    description="Tools linked to the thesis",

    license='MIT',

    # long_description=open('README.md').read(),

    # ext_modules=ext_modules,
    # Active la prise en compte du fichier MANIFEST.in
    include_package_data=True,

    entry_points={},

    # Une url qui pointe vers la page officielle de votre lib
    url='',

    # Il est d'usage de mettre quelques metadata à propos de sa lib
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers.

    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
)
