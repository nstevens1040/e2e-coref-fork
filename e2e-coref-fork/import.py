from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import json
import math
import shutil
import sys
import numpy as np
import tensorflow as tf
import pyhocon
import operator
import random
import threading
import tensorflow_hub as hub
import h5py
import util
import coref_ops
import conll
import metrics
import coref_model as cm

def cls():
    import os
    os.system('clear')


config = util.initialize_from_env()

# Input file in .jsonlines format.
input_filename = sys.argv[2]

# Predictions will be written to this file in .jsonlines format.
output_filename = sys.argv[3]

ctx=config["context_embeddings"]
default_embedding = np.zeros(ctx.size)

embedding_dict = collections.defaultdict(lambda:default_embedding)
vocab_size = None
f=open(ctx.path)
