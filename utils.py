import json
import re
import jieba
import numpy as np
import pandas as pd
import requests
import torch
from scipy.sparse import coo_matrix
from torch import nn
from pre_process.dataset_process import MyData
import zhconv


def text_to_graph(dat, dat_voc, text, context, sl=3, model=None, tokenizer=None, directed=False, sw=0, label=-1):
    """
    :param sl:          滑动窗口大小
    :param dat:         外部知识图谱
    :param dat_voc:     外部知识图谱词汇表
    :param label:       可选参数，标签
    :param text:        文本
    :param context:     上下文
    :param tokenizer:   分词器
    :param model:       词嵌入模型
    :param directed:    是否为有向图
    :param sw:          数据集处理方式（0：完整版，1：无外部知识，2：无上下文）
    :return:            图结构数据
    """
    edge_index = []
    raw_target, raw_context = text, context

    # TW, CW = ([word for word in jieba.lcut(text) if word not in stopwords],
    #          [word for word in jieba.lcut(context) if word not in stopwords])  # preprocess(text), preprocess(context)
    # text, context = zhconv.convert(text, 'zh-cn'), zhconv.convert(context, 'zh-cn')  # 繁体转简体
    """TW, CW = (
    [word for word in jieba.cut(text.replace('\t', '').replace(' ', '').replace('\n', '').replace('\u3000', '')) if
     word not in stopwords],
    [word for word in jieba.cut(context.replace('\t', '').replace(' ', '').replace('\n', '').replace('\u3000', '')) if
     word not in stopwords])  # preprocess(text), preprocess(context)"""
    voc = model.key_to_index.keys()
    TW, CW = preprocess(voc, text), preprocess(voc, context)
    CW = TW if len(CW) == 0 else CW
    TW = CW if len(TW) == 0 else TW

    """
    CW = [word for word in jieba.cut(context.replace('\t', '').replace(' ', '').replace('\n', '')) if
          word not in stopwords] if len(CW) == 0 else CW
    TW = [word for word in jieba.cut(text.replace('\t', '').replace(' ', '').replace('\n', '')) if
          word not in stopwords] if len(TW) == 0 else TW
    CW = TW if len(CW) == 0 else CW
    TW = CW if len(TW) == 0 else TW
    """
    lt, lc = len(TW), len(CW)
    target_len, context_len = lt, lc
    # St = encoding(TW, tokenizer, model)
    if len(TW) == 0:
        print(text)
        exit()
    St = embedding(model=model, word_list=TW)
    node = TW
    x = St
    KG, number = {'num': [], 'raw': []}, 0
    match sw:
        case 0:
            # Sc = encoding(CW, tokenizer, model)
            node += CW
            Sc = embedding(model=model, word_list=CW)
            x = torch.cat((x, Sc), dim=0)

            for i in TW + CW:
                n, kg = locconceptnet(i, voc, dat=dat, dat_voc=dat_voc)
                KG['num'].append(n)
                KG['raw'].append(kg)
                number += n
                if n > 0:
                    x = torch.cat((x, embedding(model, kg)), dim=0)

            KG['num'].append(number)
            edge_index = get_edge_index(lt=lt, lc=lc, sl=sl, sw=sw, KG=KG['num'], directed=directed)
        case 1:
            # Sc = encoding(CW, tokenizer, model)
            Sc = embedding(model=model, word_list=CW)
            node += CW
            x = torch.cat((St, Sc), dim=0)
            edge_index = get_edge_index(lt=lt, lc=lc, sl=sl, sw=sw, directed=directed)
        case 2:  # 待修改
            KG, number = [], 0
            for i in TW:
                n, kg = locconceptnet(i)
                KG.append(n)
                number += n
                if n > 0:
                    x = torch.cat((x, encoding(kg, tokenizer, model)), dim=0)
            KG.append(number)
            edge_index = get_edge_index(lt=lt, sw=sw, KG=KG, directed=directed)
    if label != -1:
        return MyData(raw_target=raw_target, raw_context=raw_context, ex=KG, node=node, target_len=target_len,
                      context_len=context_len, x=x, edge_index=edge_index, y=torch.tensor([label]))
    return MyData(x=x, edge_index=edge_index)


def embedding(model, word_list):
    x = torch.tensor(model[word_list])
    return x


def preprocess(voc, text):
    words = [word for word in jieba.cut(filter(text)) if word in voc]
    if len(words) == 0:
        words=[word for word in jieba.cut(text) if word in voc]
    if len(words) == 0:
        words=[word for word in text if word in voc]
    return words


"""def preprocess(text):  # 文本预处理
    with open('../datasets/inter-file/pat.dat', encoding='UTF-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    with open('../datasets/inter-file/未登录词.txt', encoding='UTF-8') as f:
        stopwords += [line.strip() for line in f.readlines()]
    text = filter(text)
    all_words = []
    all_words += [word for word in jieba.cut(text.replace('\t', '').replace(' ', '').replace('\n', '')) if
                  word not in stopwords]
    return all_words
"""


def filter(text):  # 文本过滤
    # 移除网址
    text = re.sub(r'http[s]?:[//|\\](?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+[\\/]?', ' ',
                  text)
    # 移除HTML标签
    text = re.sub('<[^>]*>', ' ', text)
    # 移除系统文本
    # text = re.sub('转发微博', ' ', text)
    # 移除html实体字符
    text = re.sub(r'&\w+;', ' ', text)
    # 移除【话题内容】
    # text = re.sub('【.*?】', ' ', text)
    # 移除#话题内容
    # text = re.sub('#.*?#', ' ', text)
    # 移除@用户
    text = re.sub('@.*?[:|\s]', ' ', text).replace('\t', '').replace(' ', '').replace('\n', '')
    # 移除空格
    # text = re.sub(r'\s', '', text)
    # 移除数字
    text = re.sub(r'-?\d+(.\d+)?', ' ', text)

    return text


def conceptnet(word):  # 返回与当前词相关的实体,调用api，速度较慢
    obj = requests.get('http://api.conceptnet.io/c/zh/' + word).json()
    bad = ['/r/Antonym', '/r/Desires', '/r/DistinctFrom', '/r/EtymologicallyDerivedFrom', '/r/EtymologicallyRelatedTo',
           '/r/ExternalURL', '/r/FormOf', '/r/HasFirstSubevent', '/r/NotDesires']
    KG = []
    for i in obj['edges']:
        if i['weight'] < 2 or i['rel']['label'] in bad:
            continue
        start, end = i['start']['label'], i['end']['label']
        if start == word and end not in KG:
            KG.append(zhconv.convert(end, 'zh-cn'))
        elif end == word and start not in KG:
            KG.append(zhconv.convert(start, 'zh-cn'))
        if len(KG) >= 10:  # 最大返回10个实体
            break
    return len(KG), KG


def is_Chinese(word):  # 修改过的
    for ch in word:
        if '\u4e00' > ch or ch > '\u9fff':
            return False
    return True


def is_incep(word):
    ex = '../datasets/inter-file/ex_已登录词.txt'
    with open(ex, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    if word in lines:
        return True
    return False


def locconceptnet(word, voc, dat, dat_voc):  # 本地返回与当前词相关的实体
    KG = []
    if word not in dat_voc:
        return 0, KG
    result = (dat[(dat['start'].str.contains(word) | dat['end'].str.contains(word)) & (dat['weights'] > 2)].
              sort_values("weights", ascending=False)).head(20)
    for i in result.index:
        i = result.loc[i]
        start = re.sub(r'(.*)\/', '', i['start'])
        end = re.sub(r'(.*)\/', '', i['end'])
        if start == word and end in voc and end not in KG:
            KG.append(end)
        elif end == word and start in voc and start not in KG:
            KG.append(start)

    return len(KG), KG


def _filcon(path):  # 用于生成需要的知识图谱
    out_file = '../datasets/inter-file/knowledge.csv'
    data = pd.read_csv(path, delimiter='\t')
    data.columns = ['uri', 'relation', 'start', 'end', 'json']
    data = data[data['start'].apply(lambda row: row.find('zh') > 0) & data['end'].apply(
        lambda row: row.find('zh') > 0)]  # 去除非中文的知识
    bad = ['/r/Antonym', '/r/Desires', '/r/DistinctFrom', '/r/EtymologicallyDerivedFrom', '/r/EtymologicallyRelatedTo',
           '/r/ExternalURL', '/r/FormOf', '/r/HasFirstSubevent', '/r/NotDesires']
    data = data[data['relation'].apply(lambda row: row not in bad)]
    """
    edge=['/r/Antonym' '/r/AtLocation' '/r/CapableOf' '/r/Causes' '/r/CausesDesire'
 '/r/DerivedFrom' '/r/Desires' '/r/DistinctFrom'
 '/r/EtymologicallyDerivedFrom' '/r/EtymologicallyRelatedTo'
 '/r/ExternalURL' '/r/FormOf' '/r/HasA' '/r/HasContext'
 '/r/HasFirstSubevent' '/r/HasProperty' '/r/HasSubevent' '/r/IsA'
 '/r/MadeOf' '/r/MotivatedByGoal' '/r/NotDesires' '/r/PartOf'
 '/r/RelatedTo' '/r/SimilarTo' '/r/SymbolOf' '/r/Synonym' '/r/UsedFor']
    """
    weights = data['json'].apply(lambda row: json.loads(row)['weight'])
    data.pop('json')
    data.insert(4, 'weights', weights)
    data.to_csv(out_file, index=False, sep='\t')


def encoding(text, tokenizer, model):  # 将文本转化为词向量
    tokens = tokenizer.encode(text, add_special_tokens=False, trncation='longest_first')
    tokens_tensor = torch.tensor([tokens])
    with torch.no_grad():
        outputs = model(tokens_tensor)
        embeddings = outputs[0][0]
    return embeddings


def get_edge_index(lt=0, lc=0, sl=3, sw=1, KG=[], directed=True):
    edge_index = []
    sl = sl + 1
    match sw:
        case 0:
            edge_index = torch.zeros((lt + lc + KG[-1], lt + lc + KG[-1]), dtype=torch.long)

            it = lt + lc
            for i in range(lt):
                edge_index[i, i + 1:(i + sl if i + sl < lt else lt)] = 1  # 滑动窗口边
                if KG[i] > 0:  # 外部知识边
                    edge_index[it:it + KG[i], i] = 1
                    it += KG[i]
            for i in range(lt, lt + lc):
                edge_index[i, i + 1:(i + sl if i + sl < lt + lc else lt + lc)] = 1  # 滑动窗口边
                if KG[i] > 0:  # 外部知识边
                    edge_index[i, it:it + KG[i]] = 1
                    it += KG[i]
                edge_index[i, :lt] = 1  # 上下文与目标句的边
        case 1:
            edge_index = torch.zeros((lt + lc, lt + lc), dtype=torch.long)
            for i in range(lt):
                edge_index[i, i + 1:(i + sl if i + sl < lt else lt)] = 1
            for i in range(lt, lt + lc):
                edge_index[i, i + 1:i + sl] = 1
                edge_index[i, :lt] = 1
        case 2:
            edge_index = torch.zeros((lt + KG[-1], lt + KG[-1]), dtype=torch.long)
            it = lt
            for i in range(lt):
                edge_index[i, i + 1:(i + sl if i + sl < lt else lt)] = 1
                if KG[i] > 0:  # 外部知识边
                    edge_index[it:it + KG[i], i] = 1
                    it += KG[i]
    if not directed:
        edge_index = edge_index + edge_index.T
    coo = coo_matrix(edge_index.numpy())
    return torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)


def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                elif method == 'normal':
                    nn.init.normal_(w)
                elif method == 'random':
                    return
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def dir_to_undir(edge, le):  # 将有向图转化为无向图
    edge_index = torch.zeros((le, le), dtype=torch.long)
    for i in range(edge.shape[1]):
        edge_index[edge[0][i], edge[1][i]] = 1
    edge_index = edge_index + edge_index.T
    coo = coo_matrix(edge_index.numpy())
    return torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
