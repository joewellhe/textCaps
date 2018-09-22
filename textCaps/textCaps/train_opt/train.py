# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from model.textcaps import TextCpasTrain, TextCapsPredict
from .dataset import Dataset
import sys
if sys.version_info.major < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')
    import codecs
    open=codecs.open


class TextClassifier(object):
    def __init__(self, model_dir, model_name, embeding_size, filter_size, num_filters):
        self.model_dir = model_dir
        self.model_name = model_name
        self.embeding_size = embeding_size
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.dropout = 0.5
        self.l2_reg_lambda = 0
        self.batch_norm = False
        self.max_text_limit_len = 5000
        self.batch_size = 32
        self.test_batch_size = 100
        self.epoches = 50
        self.optimizer = None
        self.learning_rate = 0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_batch_norm(self, batch_norm):
        self.batch_norm = batch_norm

    def set_dropout(self, dropout):
        self.dropout = dropout

    def set_l2_reg_lambda(self, l2_reg_lambda):
        self.l2_reg_lambda = l2_reg_lambda

    def set_max_text_limit_len(self, max_len):
        self.max_text_limit_len = max_len

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_test_batch_size(self, test_batch_size):
        self.test_batch_size = test_batch_size

    def set_epoches(self, epoches):
        self.epoches = epoches

    def _train(self, dataset, w2v_model_file=None, summary_dir=None):
        if w2v_model_file:
            init_w, self.embeding_size = dataset.init_embedding_weights_with_word2vec(w2v_model_file)
            print('embedding_size of word2vec:%d' % self.embeding_size)
        textcaps = TextCpasTrain(
            vocab_size=dataset.vocab_size(),
            sequence_length=dataset.sequence_length(),
            num_classes=dataset.num_classes(),
            embedding_size=self.embeding_size,
            filter_size=self.filter_size,
            num_filters=self.num_filters,
            l2_reg_lambda=self.l2_reg_lambda,
            batch_norm=self.batch_norm)
        textcaps.set_model_dir(self.model_dir, self.model_name)
        if self.optimizer is not None:
            textcaps.set_optimizer(self.optimizer)

        if self.learning_rate > 0:
            textcaps.set_learning_rate(self.learning_rate)

        if summary_dir:
           textcaps.set_summaries_dir(summary_dir)
        if w2v_model_file:
            textcaps.init_embedding_weights_with_word2vec(init_w, self.embeding_size)

        serialize_dir, _, _= textcaps.train(self.dropout, self.epoches,
                      lambda idx: dataset.next_batch(idx, self.batch_size),
                      lambda idx: dataset.next_test_batch(idx, self.test_batch_size))
        return serialize_dir

    def _accurary(self, dataset, serialize_dir):
        textcnn = TextCapsPredict(serialize_dir)
        test_set = dataset.test_batches
        test_num = dataset.test_num
        size = self.test_batch_size
        batch_num = int(test_num / size)
        last_size = test_num - batch_num * size

        all_predictions = []
        y_test = []
        for i in range(batch_num):
            start = i * size
            end = start + size
            batches = test_set[start: end]
            x_batches, y_batches = zip(*batches)
            y = textcnn.predict(x_batches)
            all_predictions = np.concatenate([all_predictions, y])
            y_batches = np.argmax(y_batches, axis=1)
            y_test = np.concatenate([y_test, y_batches])
        if last_size > 0:
            start = batch_num * size
            end = start + last_size
            x_batches, y_batches = zip(*test_set[start:end])
            y = textcnn.predict(x_batches)
            all_predictions = np.concatenate([all_predictions, y])
            y_batches = np.argmax(y_batches, axis=1)
            y_test = np.concatenate([y_test, y_batches])

        correct = sum(all_predictions == y_test)
        print('acc:%f' % (correct / test_num))

    def fit(self, trainset, vocab_file, w2v_model_file=None, min_word_freq=0, show_acc=True, summary_dir=None):
        dataset = Dataset(self.max_text_limit_len, trainset, vocab_file, min_word_freq)
        serialize_dir = self._train(dataset, w2v_model_file, summary_dir)
        # serialize_dir = 'model_bin/serialize'
        if show_acc:
            self._accurary(dataset, serialize_dir)
        return serialize_dir