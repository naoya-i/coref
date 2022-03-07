
import os

from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install


class PostProcessing(install):
    def run(self):
        install.run(self)

        os.system("bash bin/setup_all.sh")


setup(
    name="spanbertcoref",
    version="0.1.0",
    description="Coreference Resolver by SpanBERT",
    author="naoya-i",
    url="https://github.com/naoya-i/spanbertcoref",
    entry_points={'console_scripts': ['spanbertcoref=spanbertcoref:main']},
    packages=find_packages("spanbertcoref"),
    package_dir={"": "spanbertcoref"},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    install_requires=open("requirements.txt").read().splitlines(),
    include_package_data=True,
    cmdclass={'install': PostProcessing},
)
