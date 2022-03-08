import os

from setuptools import setup
from setuptools.command.install import install   
from setuptools.command.develop import develop


class PostProcessing(install):
    def run(self):
        super().run()
        compile_kernel()


class PostProcessingDev(develop):
    def run(self):
        super().run()
        compile_kernel()


def compile_kernel():
    import tensorflow as tf
    tf_extra_flags = tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    tf_extra_flags = " ".join(tf_extra_flags)

    print("Compiling kernel...")
    os.system(f"g++ -std=c++11 -shared src/coref_kernels.cc -o lib/coref_kernels.so -fPIC {tf_extra_flags} -O2 -D_GLIBCXX_USE_CXX11_ABI=1")


setup(cmdclass={
    "install": PostProcessing,
    "develop": PostProcessingDev,
})
