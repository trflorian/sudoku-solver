from setuptools import setup

setup(
    name='sudoku-solver',
    version='0.0.1',
    description='Solves a sudoku from an image',
    py_modules=['solver', 'extractor'],
    package_dir={'': 'src'},
)
