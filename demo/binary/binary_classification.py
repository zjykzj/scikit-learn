# -*- coding: utf-8 -*-

"""
@date: 2022/1/15 下午12:39
@file: binary_classification.py
@author: zj
@description: 
For the
low-precision model (two-layer neural network with 1 middle neuron) /
medium-precision model (two-layer neural network with 5 middle neurons) /
high-precision model (two-layer neural network with 10 middle neurons),
The optimal threshold of ROC curve or PR curve can improve the model performance (accuracy/precision/recall).
Relatively speaking, the lower the model performance, the more obvious the improvement effect.
对于低精度模型（二层神经网络，中间层个数为１）/中精度模型（二层神经网络，中间层个数为5）/高精度模型（二层神经网络，中间层个数为10）而言，
ROC曲线或者PR曲线的最佳阈值均能够提升模型性能（准确率/精确率/召回率），相对而言，模型性能越低，提升效果越明显；
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    confusion_matrix, ConfusionMatrixDisplay, \
    auc, roc_curve, roc_auc_score, RocCurveDisplay, \
    precision_recall_curve, average_precision_score, PrecisionRecallDisplay


def load_data():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=3, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def load_model(X, y):
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", random_state=42)
    # clf = MLPClassifier(hidden_layer_sizes=(5,), activation="relu", random_state=42)
    # clf = MLPClassifier(hidden_layer_sizes=(1,), activation="relu", random_state=42)

    model = make_pipeline(StandardScaler(), clf)
    model.fit(X, y)
    print('model type:', type(model))

    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    print('y_pred shape:', y_pred.shape)

    y_pred_prob = model.predict_proba(X_test)
    print('y_pred_prob shape:', y_pred_prob.shape)

    return y_pred, y_pred_prob


def score(y_pred, y_test):
    print('score ...')
    print('y_pred[:10]:', y_pred[:10])
    print('y_test[:10]:', y_test[:10])

    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    acc_num = accuracy_score(y_test, y_pred, normalize=False)
    print('acc:', acc, ' acc_num:', acc_num)

    # 计算精度
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_micro = precision_score(y_test, y_pred, average='micro')
    print('precision_macro:', precision_macro, ' precision_micro:', precision_micro)

    # 计算召回率
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    print('recall_macro:', recall_macro, ' recall_micro:', recall_micro)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1], normalize=None)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

    # 显示混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def score_roc(y_pred, y_test):
    print('roc curve ...')
    print('y_pred[:10]:', y_pred[:10])
    print('y_test[:10]:', y_test[:10])

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    print('tpr: {}'.format(tpr))
    print('fpr: {}'.format(fpr))
    print('thresholds: {}'.format(thresholds))

    # 计算ROC曲线下面积
    auc_score = roc_auc_score(y_test, y_pred, average="macro", multi_class='ovo')
    roc_auc = auc(fpr, tpr)
    print('auc:', auc_score, ' roc_auc:', roc_auc)

    # 显示ROC曲线
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                              estimator_name='example estimator')
    display.plot()
    plt.show()

    # 计算最佳阈值
    thresh = thresholds[np.argmax(tpr - fpr)]
    print('thresh:', thresh)

    return thresh


def score_pr(y_pred, y_test):
    print('pr curve ...')
    print('y_pred[:10]:', y_pred[:10])
    print('y_test[:10]:', y_test[:10])

    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred, pos_label=1)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('thresholds: {}'.format(thresholds))

    # 计算PR曲线下面积
    average_precision = average_precision_score(y_test, y_pred, average="macro", pos_label=1)
    print('average_precision:', average_precision)

    # 显示PR曲线
    import matplotlib.pyplot as plt
    display = PrecisionRecallDisplay(precision, recall,
                                     estimator_name='example estimator')
    display.plot()
    plt.show()

    # 计算最佳阈值
    thresh = thresholds[np.argmax(precision + recall)]
    print('thresh:', thresh)

    return thresh


def process_roc(y_pred_prob, y_test):
    thresh = score_roc(y_pred_prob[:, 1], y_test)
    # 获取最佳阈值后重新设置
    y_pred_new = y_pred_prob[:, 1] >= thresh
    assert isinstance(y_pred_new, np.ndarray)
    y_pred_new = y_pred_new.astype(int)
    score(y_pred_new, y_test)


def process_pr(y_pred_prob, y_test):
    thresh = score_pr(y_pred_prob[:, 1], y_test)
    # 选择最佳阈值进行操作
    y_pred_new = y_pred_prob[:, 1] >= thresh
    assert isinstance(y_pred_new, np.ndarray)
    y_pred_new = y_pred_new.astype(int)
    score(y_pred_new, y_test)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    model = load_model(X_train, y_train)

    y_pred, y_pred_prob = predict(model, X_test)
    score(y_pred, y_test)

    process_roc(y_pred_prob, y_test)
    # process_pr(y_pred_prob, y_test)
