# coding:utf-8
#coding:utf-8
import re
import os

import jieba
import time
import csv
import numpy as np
from train_opt.train import TextClassifier
punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")


def get_stopword(stopwords_file):
    stop_f = open(stopwords_file, "r", encoding='utf-8')
    stop_words = list()
    for line in stop_f.readlines():
        line = line.strip()
        if not len(line):
            continue

        stop_words.append(line)
    stop_f.close
    return stop_words


def process_catedir(cate_id, cate_dir, max_text_limit, is_large_text_extern, max_sample_num, stop_list):
    assert max_text_limit > 50
    b = time.time()
    train_set = []
    print('begin %s' % cate_dir)
    count = 0
    # print(list(os.walk(cate_dir)))
    for root, _, fnames in os.walk(cate_dir):

        if max_sample_num > 0 and count >= max_sample_num:
            break

        for fname in fnames:
            fpath = os.path.join(root, fname)
            count += 1
            if max_sample_num > 0 and count >= max_sample_num:
                break

            print('     processing(%d): %s' % (count, fpath))
            with open(fpath, 'r', encoding='utf-8') as rf:
                text = rf.read()
                # 去掉空白 换行 制表等
                # text = re.sub(r'\s+', '', text.strip())
                words = []
                # seg_list = jieba.cut(text, cut_all=False)
                for word in text.strip().split():
                    if word not in stop_list:
                        words.append(word)
                # words = list(text)
                # print(words)
                len_words = len(words)
                if len_words < 50:
                    continue
                elif len_words > max_text_limit:
                    if is_large_text_extern:
                        batch_size = len_words // max_text_limit
                        for i in range(0, batch_size):
                            offset = max_text_limit * i
                            data = ' '.join(words[offset: offset + max_text_limit])
                            train_set.append([cate_id, data])
                        offset = max_text_limit * batch_size
                        if offset < len_words:
                            data = ' '.join(words[offset:])
                            train_set.append([cate_id, data])
                    else:
                        train_set.append([cate_id, ' '.join(words[0: max_text_limit])])
                else:
                    train_set.append([cate_id, ' '.join(words)])

    np.random.shuffle(train_set)
    print('end %s finished, cost:%d' % (cate_dir, time.time() - b))
    return train_set


# is_cate_split_list: 指定cate_list每一项是否大文本分段
# cate_maxsize_list: 指定cate_list每一项最大样本数量限制
def process_n_cate(data_dir, cate_list, is_cate_split_list, trainset_file, max_text_limit, test_split, cate_maxsize_list, stop_list):
    cate_set_list = []
    for i, cate in enumerate(cate_list):
        cate_dir = os.path.join(data_dir, cate)
        cate_maxsize = cate_maxsize_list[i] if cate_maxsize_list else -1
        cate_set = process_catedir(i, cate_dir, max_text_limit, is_cate_split_list[i], cate_maxsize, stop_list)
        cate_set_list.append(cate_set)

    train_set = []
    test_set = []
    for i, cs in enumerate(cate_set_list):
        print('%s: %d' % (cate_list[i], len(cs)))
        n = int(len(cs) * test_split)
        test_set.extend(cs[0:n])
        train_set.extend(cs[n:])

    np.random.shuffle(test_set)
    np.random.shuffle(train_set)
    with open(trainset_file, 'w', encoding='utf-8') as wf:
        writer = csv.writer(wf, delimiter=str(','), lineterminator=str('\n'))
        writer.writerow([len(cate_list), len(test_set)])
        writer.writerows(test_set)
        writer.writerows(train_set)


def train(data_dir, cate_list, is_cate_split_list, train_param, bin_dir=None, cate_maxsize_list=None):
    if bin_dir is None:
        bin_dir = os.path.join('bin', '%d_cate' % len(cate_list))
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    print(bin_dir)
    trainset_file = os.path.join(bin_dir, 'trainset')
    vocab_file = os.path.join(bin_dir, 'vocab')
    stop_list = get_stopword("stopword")
    if not os.path.exists(trainset_file):
        process_n_cate(data_dir, cate_list, is_cate_split_list, trainset_file, train_param.max_text_limit,
                       train_param.test_split, cate_maxsize_list, stop_list)

    cf = TextClassifier(bin_dir, 'fcheck', train_param.embedding_size, train_param.filter_size, train_param.num_filters)
    cf.set_max_text_limit_len(train_param.max_text_limit)
    cf.set_batch_size(train_param.batch_size)
    cf.set_test_batch_size(train_param.test_batch_size)
    cf.set_epoches(train_param.epoches)
    cf.fit(trainset_file, vocab_file, w2v_model_file="vector/wordvec.txt", min_word_freq=train_param.min_word_freq)
    del cf
