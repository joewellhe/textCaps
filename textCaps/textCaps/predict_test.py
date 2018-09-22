import jieba
import predict as pd
import time
import os

if __name__ == "__main__":
    file = 'testtext'
    total_time = 0
    ac_count = 0
    for dir in ['testdataset/0', 'testdataset/1']:
        cata = int(dir.split('/')[1])
        for root, _, fnames in os.walk(dir):
            for fname in fnames:
                path = os.path.join(root, fname)
                with open(path, 'r', encoding='utf-8') as tf:
                    text = tf.read().strip()
                    start = time.time()
                    clssifer = pd.TextClassifier("model_bin/serialize", 'model_bin/vocab')
                    y = clssifer.predict(text)
                    if y == cata:
                        ac_count += 1
                    else:
                        print(path)
                    end = time.time()
                    total_time += (end-start)
    print('acc: % f' % (ac_count / 20.0))
    print('time: %f' % (total_time / 20.0))
