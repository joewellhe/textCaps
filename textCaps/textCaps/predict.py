# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
if sys.version_info.major < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

from model.textcaps import TextCapsPredict
from tensorflow.contrib import learn
import numpy as np
import jieba_fast as jieba


class TextClassifier(object):
    def __init__(self, serialize_dir, vocab_file):
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
            vocab_file)
        self.textcnn = TextCapsPredict(serialize_dir)

    def get_vocab_size(self):
        return len(self.vocab_processor.vocabulary_)

    def get_max_doc_len(self):
        return self.vocab_processor.max_document_length

    # words_text: 文本
    def predict(self, words_text, with_proba=False):
        words_text = ' '.join([w for w in jieba.cut(words_text)])
        if len(words_text) == 0:
            raise Exception('words_text is empty !')
        return self.predict_batch([words_text], with_proba)[0]

    # 此处未做相应改动,待调试
    def predict_batch(self, words_text_list, with_proba=False):
        if len(words_text_list) == 0:
            raise Exception('words_text_list is empty !')
        x = np.array(list(self.vocab_processor.transform(words_text_list)))
        return self.textcnn.predict_proba(
            x) if with_proba else self.textcnn.predict(x)
