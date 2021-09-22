# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:09:57 2021

@author: weixin
"""


#from training_utils import Metric
from flair.training_utils import EvaluationMetric
from flair.data import Corpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from typing import List
 
# define columns
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = r'path_for_all_the_train_test_dev_data_files'

# retrieve corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                              train_file='train.txt',
                                                              test_file='test.txt',
                                                              dev_file='dev.txt')


 
# 2. what tag do we want to predict?
tag_type = 'ner'
 
# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
 
# 4. initialize embeddings
embeddings = TransformerWordEmbeddings('bert-base-uncased')
 

 
# 5. initialize sequence tagger
from flair.models import SequenceTagger
 
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
 
# 6. initialize trainer
from flair.trainers import ModelTrainer
 
trainer: ModelTrainer = ModelTrainer(tagger, corpus)
 
# 7. start training
trainer.train('resources/taggers/example-ner',
              
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=50)
 


# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources/taggers/example-ner/weights.txt')

# print the number of Sentences in the train split
print("len train:", len(corpus.train))

# print the number of Sentences in the test split
print("len test:", len(corpus.test))

# print the number of Sentences in the dev split
print("len dev:", len(corpus.dev))

print(corpus.test[0])