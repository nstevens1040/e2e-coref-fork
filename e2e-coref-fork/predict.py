from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
from colorama import init
init()
from colorama import Fore, Back, Style
import tensorflow
import coref_model as cm
import util
import tensorflow.compat.v1 as tf
tensorflow.compat.v1.disable_v2_behavior()
print(Style.BRIGHT)

if __name__ == "__main__":
  config = util.initialize_from_env()

  # Input file in .jsonlines format.
  input_filename = sys.argv[2]

  # Predictions will be written to this file in .jsonlines format.
  output_filename = sys.argv[3]

  model = cm.CorefModel(config)
  input_file = open(input_filename).read()
  example = json.loads(input_file)
  with tf.Session() as session:
    model.restore(session)
    with open(output_filename, "w") as output_file:
      print(Fore.GREEN + "running tensorize_example")
      tensorized_example = model.tensorize_example(example, is_training=False)
      print(Fore.GREEN + "getting feed_dict")
      feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
      print(Fore.GREEN + "running model.predictions")
      _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
      print(Fore.GREEN + "running get_predicted_antecedents")
      predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
      print(Fore.GREEN + "running get_predicted_clusters")
      example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
      output_file.write(json.dumps(example))
      output_file.write("\n")

#  with tf.Session() as session:
#    model.restore(session)
#
#    with open(output_filename, "w") as output_file:
#      with open(input_filename) as input_file:
#        for example_num, line in enumerate(input_file.readlines()):
#          example = json.loads(line)
#          tensorized_example = model.tensorize_example(example, is_training=False)
#          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
#          _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
#          predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
#          example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)

#          output_file.write(json.dumps(example))
#          output_file.write("\n")
#          if example_num % 100 == 0:
#            print("Decoded {} examples.".format(example_num + 1))
