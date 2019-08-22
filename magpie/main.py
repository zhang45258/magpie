from __future__ import unicode_literals, print_function, division

import math
import os
import sys
from six import string_types
import keras
import keras.models
import numpy as np

from magpie.base.document import Document
from magpie.base.word2vec import train_word2vec, fit_scaler
from magpie.config import NN_ARCHITECTURE, BATCH_SIZE, EMBEDDING_SIZE, EPOCHS, WORD2VEC_WORKERS, MIN_WORD_COUNT, WORD2VEC_CONTEXT
from magpie.nn.input_data import get_data_for_model
from magpie.nn.models import get_nn_model
from magpie.utils import save_to_disk, load_from_disk


class Magpie(object):

    def __init__(self, keras_model=None, word2vec_model=None, scaler=None,
                 labels=None):
        self.labels = labels

        if isinstance(keras_model, string_types):
            self.load_model(keras_model)
        else:
            self.keras_model = keras_model

        if isinstance(word2vec_model, string_types):
            self.load_word2vec_model(word2vec_model)
        else:
            self.word2vec_model = word2vec_model

        if isinstance(scaler, string_types):
            self.load_scaler(scaler)
        else:
            self.scaler = scaler

    def train(self, train_dir, vocabulary, test_dir=None, callbacks=None,
              nn_model=NN_ARCHITECTURE, batch_size=BATCH_SIZE, test_ratio=0.0,
              epochs=EPOCHS, verbose=1, logdir= None, optimizer='Adam'):
        """
        Train the model on given data根据给定数据训练模型
        :param train_dir: directory with data files. Text files should end with
        '.txt' and corresponding files containing labels should end with '.lab'带有数据文件的目录。文本文件应以结尾“.txt”和包含标签的相应文件应以“.lab”结尾。
        :param vocabulary: iterable containing all considered labels可包含所有考虑的标签
        :param test_dir: directory with test files. They will be used to evaluate
        the model after every epoch of training.带有测试文件的目录。它们将用于评估每一个训练时代之后的模型。
        :param callbacks: objects passed to the Keras fit function as callbacks作为回调传递给keras fit函数的对象
        :param nn_model: string defining the NN architecture e.g. 'crnn'定义nn架构的字符串，例如“crnn”
        :param batch_size: size of one batch一批的大小
        :param test_ratio: the ratio of samples that will be withheld from training
        and used for testing. This can be overridden by test_dir.不接受培训的样本比率用于测试。这可以被test_dir覆盖。
        :param epochs: number of epochs to train要培训的时段数
        :param verbose: 0, 1 or 2. As in Keras.0、1或2。和Keras一样。

        :return: History object历史记录对象
        """

        if not self.word2vec_model:
            raise RuntimeError('word2vec model is not trained. ' + \
                               'Run train_word2vec() first.')

        if not self.scaler:
            raise RuntimeError('The scaler is not trained. ' + \
                               'Run fit_scaler() first.')

        if not os.path.isdir(train_dir):
            raise ValueError('The training directory ' + train_dir + \
                             ' does not exist')

        if test_dir and not os.path.isdir(test_dir):
            raise ValueError('The test directory ' + test_dir + \
                             ' does not exist')

        if self.keras_model:
            print('WARNING! Overwriting already trained Keras model.',
                  file=sys.stderr)

        self.labels = vocabulary
        self.keras_model = get_nn_model(
            nn_model,
            embedding=self.word2vec_model.vector_size,
            output_length=len(vocabulary),
            optimizer=optimizer  #这个参数是我从方法中提出来的
        )

        (x_train, y_train), test_data = get_data_for_model(
            train_dir,
            vocabulary,
            test_dir=test_dir,
            nn_model=self.keras_model,
            as_generator=False,
            batch_size=batch_size,
            word2vec_model=self.word2vec_model,
            scaler=self.scaler,
        )
        keras_model = self.keras_model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=test_data,
            validation_split=test_ratio,
            callbacks=callbacks or [],
            verbose=verbose,
        )
        #logdir保存日志的文件位置
        if logdir:
            with open(logdir, 'w') as f:
                f.write(str(keras_model.history))
        return keras_model

    def batch_train(self, train_dir, vocabulary, test_dir=None, callbacks=None,
                    nn_model=NN_ARCHITECTURE, batch_size=BATCH_SIZE,
                    epochs=EPOCHS, verbose=1):
        """
        Train the model on given data
        :param train_dir: directory with data files. Text files should end with
        '.txt' and corresponding files containing labels should end with '.lab'
        :param vocabulary: iterable containing all considered labels
        :param test_dir: directory with test files. They will be used to evaluate
        the model after every epoch of training.
        :param callbacks: objects passed to the Keras fit function as callbacks
        :param nn_model: string defining the NN architecture e.g. 'crnn'
        :param batch_size: size of one batch
        :param epochs: number of epochs to train
        :param verbose: 0, 1 or 2. As in Keras.

        :return: History object

        根据给定数据训练模型

：param train_dir：带有数据文件的目录。文本文件应以结尾

“.txt”和包含标签的相应文件应以“.lab”结尾。

：参数词汇：可包含所有考虑的标签

：param test_dir：带有测试文件的目录。它们将用于评估

每一个训练时代之后的模型。

：param callbacks：作为回调传递给keras fit函数的对象

：param nn_model：定义nn架构的字符串，例如“crnn”

：param batch_size：一批的大小

：param epochs：要培训的时段数

：param verbose:0、1或2。和Keras一样。



：返回：历史记录对象


        """

        if not self.word2vec_model:
            raise RuntimeError('word2vec model is not trained. ' + \
                               'Run train_word2vec() first.')

        if not self.scaler:
            raise RuntimeError('The scaler is not trained. ' + \
                               'Run fit_scaler() first.')

        if not os.path.isdir(train_dir):
            raise ValueError('The training directory ' + train_dir + \
                             ' does not exist')

        if test_dir and not os.path.isdir(test_dir):
            raise ValueError('The test directory ' + test_dir + \
                             ' does not exist')

        if self.keras_model:
            print('WARNING! Overwriting already trained Keras model.',
                  file=sys.stderr)

        self.labels = vocabulary
        self.keras_model = get_nn_model(
            nn_model,
            embedding=self.word2vec_model.vector_size,
            output_length=len(vocabulary)
        )

        train_generator, test_data = get_data_for_model(
            train_dir,
            vocabulary,
            test_dir=test_dir,
            nn_model=self.keras_model,
            as_generator=True,
            batch_size=batch_size,
            word2vec_model=self.word2vec_model,
            scaler=self.scaler,
        )

        nb_of_files = len({filename[:-4] for filename in os.listdir(train_dir)})
        steps_per_epoch = math.ceil(nb_of_files / batch_size)

        return self.keras_model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=test_data,
            callbacks=callbacks or [],
            verbose=verbose,
        )

    def predict_from_file(self, filepath):
        """
        Predict labels for a txt file
        :param filepath: path to the file

        :return: list of labels with corresponding confidence intervals
        预测TXT文件的标签

：param file path：文件路径



：返回：具有相应置信区间的标签列表
        """
        doc = Document(0, filepath)
        return self._predict(doc)

    def predict_from_text(self, text):
        """
        Predict labels for a given string of text
        :param text: string or unicode with the text
        :return: list of labels with corresponding confidence intervals
        预测给定文本字符串的标签

：param text:string或带文本的unicode

：返回：具有相应置信区间的标签列表
        """
        doc = Document(0, None, text=text)
        return self._predict(doc)

    def _predict(self, doc):
        """
        Predict labels for a given Document object
        :param doc: Document object
        :return: list of labels with corresponding confidence intervals
        预测给定文档对象的标签

：param doc:document对象

：返回：具有相应置信区间的标签列表
        """
        if type(self.keras_model.input) == list:
            _, sample_length, embedding_size = self.keras_model.input_shape[0]
        else:
            _, sample_length, embedding_size = self.keras_model.input_shape

        words = doc.get_all_words()[:sample_length]
        x_matrix = np.zeros((1, sample_length, embedding_size))

        for i, w in enumerate(words):
            if w in self.word2vec_model:
                word_vector = self.word2vec_model[w].reshape(1, -1)
                scaled_vector = self.scaler.transform(word_vector, copy=True)[0]
                x_matrix[doc.doc_id][i] = scaled_vector

        if type(self.keras_model.input) == list:
            x = [x_matrix] * len(self.keras_model.input)
        else:
            x = [x_matrix]

        y_predicted = self.keras_model.predict(x)

        zipped = zip(self.labels, y_predicted[0])

        return sorted(zipped, key=lambda elem: elem[1], reverse=True)

    def init_word_vectors(self, train_dir, vec_dim=EMBEDDING_SIZE):
        """
        Train word2vec model and fit the scaler afterwards
        :param train_dir: directory with '.txt' files
        :param vec_dim: dimensionality of the word vectors

        :return: None
        培训word2vec模型，然后数据预处理

：param train_dir:directory和“.txt”文件

：param vec_dim：字向量的维数



返回：没有
        """
        self.train_word2vec(train_dir, vec_dim=vec_dim)
        self.fit_scaler(train_dir)

    def train_word2vec(self, train_dir, wks=WORD2VEC_WORKERS, vec_dim=EMBEDDING_SIZE, MWC=MIN_WORD_COUNT, w2vc=WORD2VEC_CONTEXT):
        """
        Train the word2vec model on a directory with text files.
        :param train_dir: directory with '.txt' files
        :param vec_dim: dimensionality of the word vectors

        :return: trained gensim model
        在包含文本文件的目录上训练word2vec模型。

：param train_dir:“.txt”文件的目录

：param vec_dim：字向量的维数



：返回：培训的Gensim模型
        """
        if self.word2vec_model:
            print('WARNING! Overwriting already trained word2vec model.',
                  file=sys.stderr)

        self.word2vec_model = train_word2vec(train_dir, wks=wks, vec_dim=vec_dim, MWC=MWC, w2vc=w2vc)

        return self.word2vec_model

    def fit_scaler(self, train_dir):
        """
        Fit a scaler on given data. Word vectors must be trained already.
        :param train_dir: directory with '.txt' files

        :return: fitted scaler object
        在给定的数据上安装一个定标器。单词向量必须已经过训练。

：param train_dir:directory和“.txt”文件



：返回：已安装的定标器对象
        """
        if not self.word2vec_model:
            raise ValueError('word2vec model is not trained. ' + \
                             'Run train_word2vec() first.')

        if self.scaler:
            print('WARNING! Overwriting already fitted scaler.',
                  file=sys.stderr)

        self.scaler = fit_scaler(train_dir, word2vec_model=self.word2vec_model)

        return self.scaler

    def save_scaler(self, filepath, overwrite=False):
        """ Save the scaler object to a file """
        if not self.scaler:
            raise ValueError("Can't save the scaler, " + \
                             "it has not been trained yet")
        save_to_disk(filepath, self.scaler, overwrite=overwrite)

    def load_scaler(self, filepath):
        """ Load the scaler object from a file """
        self.scaler = load_from_disk(filepath)

    def save_word2vec_model(self, filepath, overwrite=False):
        """ Save the word2vec model to a file """
        if not self.word2vec_model:
            raise ValueError("Can't save the word2vec model, " + \
                             "it has not been trained yet")
        save_to_disk(filepath, self.word2vec_model, overwrite=overwrite)

    def load_word2vec_model(self, filepath):
        """ Load the word2vec model from a file """
        self.word2vec_model = load_from_disk(filepath)

    def save_model(self, filepath):
        """ Save the keras NN model to a HDF5 file """
        if not self.keras_model:
            raise ValueError("Can't save the model, " \
                             "it has not been trained yet")

        if os.path.exists(filepath):
            raise ValueError("File " + filepath + " already exists!")
        self.keras_model.save(filepath)

    def load_model(self, filepath):
        """ Load the keras NN model from a HDF5 file """
        if not os.path.exists(filepath):
            raise ValueError("File " + filepath + " does not exist")
        self.keras_model = keras.models.load_model(filepath)


