import importlib_resources
import tensorflow as tf


RESOURCE_ROOT = importlib_resources.files("spanbertcoref")


coref_op_library = tf.load_op_library(str(RESOURCE_ROOT / "lib" / "coref_kernels.so"))

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")
