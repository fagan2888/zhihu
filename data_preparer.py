# -*- coding: utf-8 -*-

import gc

import numpy as np
import pandas as pd

import config
from utils import pkl_utils
from utils import help_function


def main():
	dfLabel = pd.read_csv(config.TRAIN_DATA_LABEL, sep='\t', names=['qid', 'tids'])
	dfTrain = pd.read_csv(config.TRAIN_DATA, sep='\t', names=['qid', 'q_title_chars', 'q_title_words', 'q_desc_chars', 'q_desc_words'])
	dfTest = pd.read_csv(config.TEST_DATA, sep='\t', names=['qid', 'q_title_chars', 'q_title_words', 'q_desc_chars', 'q_desc_words'])
	dfTopic = pd.read_csv(config.TOPIC_DATA, sep='\t', names=['tid', 'ptids', 't_title_chars', 't_title_words', 't_desc_chars', 't_desc_words'])
	dfCharsEmbeddings = pd.read_csv(config.CHAR_EMBEDDING, sep='\t', names=['cid', 'embeddings'])
	dfWordsEmbeddings = pd.read_csv(config.WORD_EMBEDDING, sep='\t', names=['wid', 'embeddings'])

	#
	print("Lenght of Train set %d" % len(dfTrain))
	print("Length of Topic %d" % len(dfTopic))
	print("Length of Test set %d" % len(dfTest))

	# expload labels
	dfTrain['tids'] = dfLabel['tids'].values
	dfTrain['tids'] = dfTrain['tids'].str.split(',')
	df = help_function.pandas_explode(dfTrain, 'tids')
	df.fillna(value="", inplace=True)


	print("Length of explode Train set %d" % len(df))
	dfAll = pd.concat((df, dfTest))

	print df.head()

	# dfAll = pd.concat((dfTrain, dfTest))


if __name__ == "__main__":
	main()