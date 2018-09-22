from train_opt.train import TextClassifier
import os


class TrainParam(object):
    def __init__(self):
        self.max_text_limit = 800
        self.embedding_size = 100
        self.filter_size = 3
        self.num_filters = 45
        self.min_word_freq = 5
        self.epoches = 50
        self.batch_size = 32
        self.test_batch_size = 100


bin_dir = 'model_bin'
trainset_file='data/trainset'
vocab_file=os.path.join(bin_dir, 'vocab')
if not os.path.exists(bin_dir):
    os.makedirs(bin_dir)
train_param = TrainParam()
cf = TextClassifier(bin_dir, 'fcheck', train_param.embedding_size, train_param.filter_size, train_param.num_filters)
cf.set_max_text_limit_len(train_param.max_text_limit)
cf.set_batch_size(train_param.batch_size)
cf.set_test_batch_size(train_param.test_batch_size)
cf.set_epoches(train_param.epoches)
cf.fit(trainset_file, vocab_file, w2v_model_file="vector/wordvec.txt", min_word_freq=train_param.min_word_freq)
del cf
