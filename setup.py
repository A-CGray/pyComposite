from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('pyComposite/__init__.py').read(),
)[0]

setup(name='pyComposite',
      version=__version__,


      description="pyComposite is an object oriented framework for basic analysis of composite laminate plates",
      keywords='Composites',
      author='Alasdair Christison Gray',
      author_email='',
      url='https://github.com/A-Gray-94/pyComposite',
      license='Apache License Version 2.0',
      packages=[
          'pyComposite',
      ],
      install_requires=[
            'numpy'

      ],
      classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python"]
      )