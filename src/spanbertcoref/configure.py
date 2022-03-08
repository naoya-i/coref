import importlib_resources
import os


def compile_kernel():
    import tensorflow as tf
    tf_extra_flags = tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    tf_extra_flags = " ".join(tf_extra_flags)

    print("Compiling kernel...")
    
    cc_file = importlib_resources.files("spanbertcoref") / "lib" / "coref_kernels.cc"
    so_file = importlib_resources.files("spanbertcoref") / "lib" / "coref_kernels.so"

    os.system(f"g++ -std=c++11 -shared {cc_file} -o {so_file} -fPIC {tf_extra_flags} -O2 -D_GLIBCXX_USE_CXX11_ABI=1")


def main():
    compile_kernel()

main()