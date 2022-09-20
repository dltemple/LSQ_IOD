from distutils.core import setup
from Cython.Build import cythonize

setup(author='Dwight Temple',
        ext_modules = cythonize("SOFA_cy.pyx")
)