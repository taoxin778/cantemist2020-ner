import spacy
import os
import pandas as pd
import pickle
import glob
import re
from collections import Counter
import matplotlib.pyplot as plt

nlp = spacy.load("es_core_news_sm")


def get_doc_name(file_path):
    train_doc = [os.path.basename(f).strip('.txt')
                 for f in glob.glob(os.path.join(file_path, '*.txt'))]
    return train_doc


def get_labels(tag_schems: str):
    labels = []
    if tag_schems == "bio":
        labels = ['B', 'I', "O"]

    if tag_schems == "BIOMS":
        labels = ["B", "I", "O", "M", "S"]

    return labels


def read_raw_text(file_path: str, doc_list: list, islabel: bool):
    ids = doc_list
    texts = []
    annotations = []
    for doc in doc_list:
        path = os.path.join(file_path, doc) + ".txt"
        with open(path, 'r', encoding='utf-8-sig') as file:
            text = file.read()
        s_text = nlp(text)
        texts.append(s_text)
        if islabel:
            xml_path = os.path.join(file_path, doc) + ".ann"
            ann = extracte_annotation(xml_path)
            annotations.append(ann)
    return [ids, texts, annotations]


def extracte_annotation(file_path):
    with open(file_path, "r", encoding="utf-8-sig")as file:
        ann_lines = file.readlines()
        return ann_lines


def splite_ann(ann: str):
    entity_id, span, content = ann.split(sep="\t")
    span_f = span.split()[1]
    span_s = span.split()[2]
    span_pair = (int(span_f), int(span_s))
    content = content.split()
    return span_pair, content


def data_static(data):

    import matplotlib as plt

    doc_len = list(map(len, data[1]))
    doc_len_counter = Counter(doc_len)
    ann_len = list(map(len, data[2]))
    ann_len_counter = Counter(ann_len)

    print(doc_len_counter.most_common())
    print("#" * 10)
    print(ann_len_counter.most_common())



def is_tittle_sent(sent):
    '''
    判断是否为标题
    :param sent:
    :return:
    '''
    label = True
    for tk in sent:
        if not tk.is_title:
            return False
    return label


def is_clean_punctuation(token):
    if token.is_punct or token.text == "\n" or token.is_space:
        return True
    else:
        return False


def sort_ann(anns):
    anns = [splite_ann(ann) for ann in anns]
    anns.sort(key=lambda ann: ann[0][0])
    return anns


def get_ann_label(ann_idx, ann):
    try:
        span_pair, contex = ann[ann_idx]
        span_1 = span_pair[0]
        span_2 = span_pair[1]
    except IndexError:
        span_1 = 0
        span_2 = -1
    return span_1, span_2

def resub_num(token):
    if token.like_num:
        return "Cifra"
    else:
        return token.text


def split_and_tag(data):
    tag_labels = get_labels("bio")
    ids = data[0]
    docs = data[1]
    anns = data[2]
    sents_lables = []

    for id, doc, ann in zip(ids, docs, anns):
        first = True
        ann_idx = 0
        ann = sort_ann(ann)
        for sent in doc.sents:
            if is_tittle_sent(sent):
                continue
            sp_sent = []
            label = []
            text = []
            for tok in sent:
                if is_clean_punctuation(tok):
                    continue
                te = resub_num(tok)
                left, right = get_ann_label(ann_idx, ann)
                if left <= tok.idx < right:
                    # print(tok.idx)
                    sp_sent.append(tok)
                    text.append(te)
                    if first:
                        label.append(tag_labels[0])  # append B
                        first = False
                    else:
                        label.append(tag_labels[1])  # append I
                    if doc[tok.i + 1].idx >= right:
                        ann_idx += 1
                        first = True
                else:
                    sp_sent.append(tok)
                    text.append(te)
                    label.append(tag_labels[2])
            sents_lables.append((id, sp_sent, text, label))
    return sents_lables



def splite_test(data):
    tag_labels = get_labels("bio")
    ids = data[0]
    docs = data[1]
    sents = []
    for id, doc in zip(ids, docs):
        for sent in doc.sents:
            if is_tittle_sent(sent):
                continue
            sp_sent = []
            label = []
            text = []
            for tok in sent:
                if is_clean_punctuation(tok):
                    continue
                te = resub_num(tok)
                sp_sent.append(tok)
                text.append(te)
                label.append(tag_labels[2])
            sents.append((id, sp_sent, text, label))
    return sents




def read_data_ner(file, mode):
    path = os.path.join(file, mode + "/")
    file_list = get_doc_name(path)
    data = read_raw_text(path, file_list, islabel=True if mode != "test" else False)
    tag_data = None
    if mode != "test":
        tag_data = split_and_tag(data)
    else:
        tag_data = splite_test(data)
    return tag_data


if __name__ == '__main__':
    data = read_data_ner("./demo/", "test")
    data_static(data)
    # sents = data[1]
    # sents = [d[1] for d in data]
    # len_sent = map(len, sents)
    # len_counter = Counter(len_sent)
    # # print(len_counter.most_common(20))
    # keys = len_counter.keys()
    # val = len_counter.values()
    # plt.bar(keys, val, label=keys)
    # plt.show()