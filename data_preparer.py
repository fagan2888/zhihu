# -*- coding: utf-8 -*-

import gc

import numpy as np
import pandas as pd

import config
from utils import pkl_utils

def main():
	dfLabel = pd.read_csv(config.TRAIN_DATA_LABEL, sep='\t', names=['qid', 'tids'])
	dfTrain = pd.read_csv(config.TRAIN_DATA, sep='\t', names=['qid', 'chars', 'words'])
	dfTest = pd.read_csv(config.TEST_DATA, sep='\t', names=['qid', 'chars', 'words'])
	dfTopic = pd.read_csv(config.TOPIC_DATA, sep='\t', names=['tid', 'ptids', 'title_chars', 'title_words', 'desc_chars', 'desc_words'])
	dfCharsEmbeddings = pd.read_csv(config.CHAR_EMBEDDING, sep='\t', names=['cid', 'embeddings'])
	dfWordsEmbeddings = pd.read_csv(config.WORD_EMBEDDING, sep='\t', names=['wid', 'embeddings'])

	#
	print("Lenght of Train set %d" % len(dfTrain))
	print("Length of Topic %d" % len(dfTopic))
	print("Length of Test set %d" % len(dfTest))

	#
	dfTrain['tids'] = dfLabel['tids']
	print dfTrain.head()

	# dfAll = pd.concat((dfTrain, dfTest))


if __name__ == "__main__":
	main()