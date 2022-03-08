import importlib_resources
import os


RESOURCE_ROOT = importlib_resources.files("spanbertcoref")


def compile_kernel():
    import tensorflow as tf
    tf_extra_flags = tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    tf_extra_flags = " ".join(tf_extra_flags)

    print("Compiling kernel...")
    
    cc_file = RESOURCE_ROOT / "lib" / "coref_kernels.cc"
    so_file = RESOURCE_ROOT / "lib" / "coref_kernels.so"

    os.system(f"g++ -std=c++11 -shared {cc_file} -o {so_file} -fPIC {tf_extra_flags} -O2")


def maybe_copy_conf():
    origin_conf_file = RESOURCE_ROOT / "template.conf"
    conf_file = os.path.expanduser("~/.spanbertcoref.conf")

    if not os.path.exists(conf_file):
        print("Copying conf file...")
        os.system(f"cp {origin_conf_file} {conf_file}")


def main():
    compile_kernel()
    maybe_copy_conf()


main()