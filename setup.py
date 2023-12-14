"""

"""

from setuptools import setup
from setuptools import find_packages

VERSION:str = "0.1.0"
LICENSE:str = "MIT"
DESCRIPTION:str = "camera video noiseprint extractor"
NAME:str = "noiseprint"
AUTHOR:str = "hamzeh asadi"
EMAIL:str = "myemail@mail.com"


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=open("README.md").read(),
    license=LICENSE,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(),
    requires=["torchvison", "av"]
)
