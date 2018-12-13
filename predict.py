from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  log_dir = config["log_dir"]

  # Input file in .jsonlines format.
  input_filename = sys.argv[2]

  # Predictions will be written to this file in .jsonlines format.
  output_filename = sys.argv[3]

  model = cm.CorefModel(config)
  saver = tf.train.Saver()

  with tf.Session() as session:
    model.restore(session)
    # ckpt = tf.train.get_checkpoint_state(log_dir)
    # if ckpt and ckpt.model_checkpoint_path:
      # print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      # saver.restore(session, ckpt.model_checkpoint_path)

    with open(output_filename, "w") as output_file:
      with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
          example = json.loads(line)
          tensorized_example = model.tensorize_example(example, is_training=False)
          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
          _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
          predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
          example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
          example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
          example['head_scores'] = []

          output_file.write(json.dumps(example))
          output_file.write("\n")
          if example_num % 100 == 0:
            print("Decoded {} examples.".format(example_num + 1))
