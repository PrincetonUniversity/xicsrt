import setuptools

with open('README') as ff:
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
    ,'description':'A simple raytracing application written in Python.'
    ,'long_description':long_description
    ,'url':'http://amicitas.bitbucket.org/xicsrt/'
    ,'license':'MIT'
    ,'packages':setuptools.find_packages()
    ,'classifiers':classifiers
    ,'install_requires':['numpy', 'pillow']
    ,'python_requires':'>=3.6'
    }
    
setuptools.setup(**params)
