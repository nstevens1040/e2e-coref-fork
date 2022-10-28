from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import input
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import coref_model as cm
import util
import json
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import sys

def create_example(text):
  raw_sentences = sent_tokenize(text)
  sentences = [word_tokenize(s) for s in raw_sentences]
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }

def print_predictions(example):
  words = util.flatten(example["sentences"])
  for cluster in example["predicted_clusters"]:
    print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))

def get_predictions(example,output_filename):
    ra = []
    words = util.flatten(example["sentences"])
    for cluster in example["predicted_clusters"]:
      ra.append("Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))
      print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))
    str_predictions = "\n".join(ra)
    with open(output_filename,"w") as ff:
      ff.write(str_predictions)
    print("\nwrote text file:" + output_filename + "\n")


def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)
  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

  example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["top_spans_list"] = list(example["top_spans"])
  example["head_scores"] = head_scores.tolist()
  return example

if __name__ == "__main__":
  input_realpath = os.getcwd() + "/" + sys.argv[2]
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  with tf.Session() as session:
    model.restore(session)
    while True:
#      text = input("Document text: ")
      text = open(input_realpath).read()
      if len(text) > 0:
        exa = make_predictions(text, model)
        next_obj = {}
        next_obj["doc_key"] = exa["doc_key"]
        next_obj["clusters"] = exa["clusters"]
        next_obj["sentences"] = exa["sentences"]
        next_obj["speakers"] = exa["speakers"]
        next_obj["predicted_clusters"] = exa["predicted_clusters"]
        next_obj["head_scores"] = exa["head_scores"]
        next_obj["top_spans"] = exa["top_spans_list"]
        json_file = os.getcwd() + "/results/" + os.path.splitext(os.path.basename(input_realpath))[0] + ".json"
        text_output = os.getcwd() + "/results/" + os.path.splitext(os.path.basename(input_realpath))[0] + ".txt"
        with open(json_file,"w") as ff:
          json.dump(next_obj,ff)
        print("\nwrote json file\n")
        get_predictions(exa,text_output)
        print_predictions(exa)
