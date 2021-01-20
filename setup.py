"""
To submit an update to pypi:
  1. Update _version.py
  2. python setup.py sdist bdist_wheel
  3. python -m twine upload --repository testpypi dist/*
  4. python -m twine upload dist/*
"""

import setuptools

with open('README.md') as ff:
    long_description = ff.read()

exec(open('xicsrt/_version.py').read())

classifiers = [
    "Development Status :: 4 - Beta"
    ,"License :: OSI Approved :: MIT License"
    ,"Programming Language :: Python :: 3"
    ,"Operating System :: OS Independent"
    ]

params ={
    'name':'xicsrt'
    ,'version':__version__
    ,'author':'Novimir Antoniuk Pablant'
    ,'author_email':'npablant@pppl.gov'
    ,'description':'A photon based raytracing application written in Python.'
    ,'long_description':long_description
    ,'long_description_content_type':'text/markdown'
    ,'url':'http://amicitas.bitbucket.org/xicsrt'
    ,'license':'MIT'
    ,'packages':setuptools.find_packages()
    ,'classifiers':classifiers
    ,'install_requires':['numpy', 'scipy', 'pillow']
    ,'python_requires':'>=3.8'
    }
    
setuptools.setup(**params)
