import os.path

import pandas as pd
from gensim import models
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data


class MyData(Data):
    def __init__(self, raw_target=None, raw_context=None, ex=None, node=None, target_len=None, context_len=None,
                 embedding=None, **kwargs):
        """

        :param raw_target:  原始目标句
        :param raw_context: 原始上下文
        :param ex:          原始外部知识
        :param node:        节点文本形式
        :param target_len:  目标句节点长度
        :param context_len: 上下文节点长度
        :param embedding:   节点嵌入方式
        :param kwargs:      其他参数
        """
        super().__init__(**kwargs)
        self.raw_target = raw_target
        self.raw_context = raw_context
        self.ex = ex
        self.node = node
        self.target_len = target_len
        self.context_len = context_len
        self.embedding = embedding


class MyDataset(Dataset):  # 小内存方案
    def __init__(self, root='../datasets', transform=None, pre_transform=None, pre_filter=None,
                 file_name='train.tsv', dataset='smp2019-ecisa', sw=0, sl=3,directed=True):
        """
        :param root:            数据集根目录
        :param transform:       每一次数据加载到程序之前都会默认调用进行数据转换
        :param pre_transform:   返回转换后的版本，在数据被存储到硬盘之前进行转换（只发生一次）
        :param file_name:       文件名称（默认为训练集）
        :param dataset:         数据集名称（默认为smp2019-ecisa）
        :param sw:              数据集处理方式（0：完整版，1：无外部知识，2：无上下文）
        :param sl:              滑动窗口大小（默认为3）
        :param directed:        是否为有向图（默认为无向图）
        """
        self.file_name = file_name
        self.processed_file = file_name.replace('.tsv', ('_directed.pt' if directed else '_undirected.pt')) \
            .replace('.pt', '_sw{}_sl{}'.format(sw,sl))
        self.sw = sw
        self.sl = sl
        self.directed = directed
        self.dataset = dataset
        with open(os.path.join(root + '/new/' + dataset, 'labels.json'), 'r', encoding='utf-8') as f:
            lab=f.readlines()
        self.num_class=len(lab)
        with open(os.path.join(root + '/new/' + dataset, file_name), 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # 跳过标题行
        self.le = len(lines)  # 数据集的长度
        super(MyDataset, self).__init__(root, transform, pre_transform, pre_filter)

    def len(self) -> int:
        return self.le

    def get(self, idx: int):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    @property
    def raw_dir(self) -> str:  # 返回原始数据集的根目录
        return os.path.join(self.root+'/new', self.dataset)

    @property
    def processed_dir(self) -> str:  # 返回处理后的数据集的根目录
        return os.path.join(self.root + '/processed/' + self.dataset, self.processed_file)

    @property
    def raw_file_names(self):  # 返回原始数据集中的文件名
        return [self.file_name]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.le)]

    def download(self):  # 不需要下载
        pass

    def process(self):
        idx = 0

        # model_name = 'hfl/minirbt-h288'
        # tokenizer = BertTokenizer.from_pretrained(model_name)
        # model = BertModel.from_pretrained(model_name)
        path = self.root.replace('datasets', '') + 'model/pre_train/word2vec/merge_sgns_bigram_char300.txt'
        # path=self.root.replace('datasets','')+'model/pre_train/word2vec/tencent-ailab-embedding.txt'
        model = models.KeyedVectors.load_word2vec_format(path, binary=False, encoding='utf-8', unicode_errors='ignore')

        file = self.root.replace('datasets', '') + 'datasets/inter-file/knowledge.csv'
        dat = pd.read_csv(file, delimiter='\t')
        ex = os.path.join(self.root, 'inter-file/ex_词库.txt')

        with open(ex, 'r', encoding='utf-8') as f:
            dat_voc = [line.strip() for line in f.readlines()]

        with open(os.path.join(self.raw_dir, self.file_name), 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # 跳过标题行

        for line in tqdm(lines, colour='green', desc='数据集处理中'):
            if os.path.exists(os.path.join(self.processed_dir, f'data_{idx}.pt')):
                idx += 1
                continue

            from utils import text_to_graph
            line = line.strip().split('\t')
            label, context, text = int(line[0]), line[1], line[2]
            data = text_to_graph(dat=dat, dat_voc=dat_voc, text=text, sl=self.sl,context=context, model=model,
                                 directed=self.directed, sw=self.sw,
                                 label=label)
            if not data.validate():
                print(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1


class InMyDataset(InMemoryDataset):  # 继承InMemoryDataset类，将数据集全部读入内存,存在问题:len()函数返回1，slices为None
    def __init__(self, root='../datasets', transform=None, pre_transform=None, pre_filter=None, file_name='train.tsv',
                 sw=0, directed=False):
        """
        :param root:            数据集根目录
        :param transform:       每一次数据加载到程序之前都会默认调用进行数据转换
        :param pre_transform:   返回转换后的版本，在数据被存储到硬盘之前进行转换（只发生一次）
        :param file_name:       数据集名称（默认为训练集）
        :param sw:              数据集处理方式（0：完整版，1：无外部知识，2：无上下文）
        :param directed:        是否为有向图（默认为无向图）
        """
        self.file_name = file_name
        self.processed_file = file_name.replace('.tsv', ('_directed.pt' if directed else '_undirected.pt')) \
            .replace('.pt', '_sw{}.pt'.format(sw))
        self.sw = sw
        self.directed = directed
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'smp2019-ecisa')

    @property
    def processed_dir(self) -> str:  # 返回处理后的数据集的根目录
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):  # 返回原始数据集中的文件名
        return [self.file_name]

    @property
    def processed_file_names(self):
        return [self.processed_file]

    def download(self):
        pass

    def process(self):
        import utils
        datalist = []
        model_name = 'hfl/minirbt-h288'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        with open(os.path.join(self.raw_dir, self.file_name), 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # 跳过标题行
        for line in tqdm(lines):
            line = line.strip().split('\t')
            label, context, text = int(line[0]), line[1], line[2]
            datalist.append(utils.text_to_graph(text, context, tokenizer, model, directed=self.directed, sw=self.sw,
                                                label=label))
        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])
