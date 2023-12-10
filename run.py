import argparse
import json
import os
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from eval import train, test
from model.TGR import TGR
from pre_process.dataset_process import MyDataset
from utils import init_network

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--dataset', default='smp2019-ecisa', type=str, help='smp2019-ecisa,NLPCC2014-SC,dataset root')
parser.add_argument('--type', default=0, type=int, help='0,1,2 dataset process type')
parser.add_argument('--slide',default=3,type=int,help='slide window size')
parser.add_argument('--direct', default=True, type=bool, help='directed or undirected')
parser.add_argument('--in_channels', default=300, type=int, help='input channels')
parser.add_argument('--hidden_channels', default=768, type=int, help='hidden channels')
parser.add_argument('--num_heads', default=3, type=int, help='num heads')
parser.add_argument('--layers', default=3, type=int, help='layers')
parser.add_argument('--dropout', default=0.4, type=float, help='dropout')
parser.add_argument('--pooling', default=0, type=int, help='pooling type')
parser.add_argument('--conv', default='GAT', type=str, help='conv type')
parser.add_argument('--lr', default=6e-6, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='weight decay')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--init', default='random', type=str, help='xavier,kaiming,normal,random init weight')
parser.add_argument('--epochs', default=300, type=int, help='epochs')
parser.add_argument('--patience', default=10, type=int, help='patience')
parser.add_argument('--seed', default=114514, type=int, help='seed')
parser.add_argument('--device', default='None', type=str, help='device')
parser.add_argument('--sep',default='',type=str,help='sep note')
args = parser.parse_args()


def init_log():
    return {
        'dataset': args.dataset,
        'type': args.type,
        'slide': args.slide,
        'direct': args.direct,
        'in_channels': args.in_channels,
        'hidden_channels': args.hidden_channels,

        'num_heads': args.num_heads,
        'layers': args.layers,
        'dropout': args.dropout,
        'pooling': args.pooling,
        'conv': args.conv,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'init': args.init,
        'epochs': args.epochs,
        'patience': args.patience,
        'seed': args.seed,
    }

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':

    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    set_seed(args.seed)

    in_channels = args.in_channels
    hidden_channels = args.hidden_channels
#    num_class = args.num_class
    num_heads = args.num_heads
    layers = args.layers
    dropout = args.dropout
    pooling = args.pooling
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    EPOCH = args.epochs
    patience = args.patience
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    log_dir = 'model/log/{}'.format(args.dataset)
    root = 'datasets'
    log_path = os.path.join(log_dir, 'log')

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = log_dir + '/best_model_{}_{}.pth'.format(args.type, 'directed' if args.direct else 'undirected')
    data_save_path = log_dir + '/best_model_{}_{}.json'.format(args.type, 'directed' if args.direct else 'undirected')

    train_dataset = MyDataset(root=root, dataset=args.dataset, file_name='train.tsv', sw=args.type,
                              directed=args.direct)
    dev_dataset = MyDataset(root=root, dataset=args.dataset, file_name='dev.tsv', sw=args.type,sl=args.slide, directed=args.direct)
    test_dataset = MyDataset(root=root, dataset=args.dataset, file_name='test.tsv', sw=args.type,sl=args.slide, directed=args.direct)

    num_class= train_dataset.num_class

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    model = TGR(in_channels=in_channels, hidden_channels=hidden_channels, num_heads=num_heads, layers=layers,
                dropout=dropout, pooling=pooling, num_classes=num_class, conv=args.conv)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = torch.nn.CrossEntropyLoss()
    loss.to(device)

    init_network(model, method=args.init)
    model.to(device)

    # 绘图数据
    x, train_loss, dev_loss = [], [], []

    if os.path.exists(data_save_path):
        with open(data_save_path, 'r', encoding='utf-8') as f:
            best = json.load(f)
    else:
        best = {'performance': {'acc': 0, 'f1': 0, 'r': 0, 'p': 0}, 'config': init_log()}
    count = 0
    dl = {'acc': 0, 'f1': 0, 'r': 0, 'p': 0, 'loss': 0}
    start_time = time.time()

    bl = 1
    for epoch in range(EPOCH):
        torch.cuda.empty_cache()    #清除显存

        acc, f1, r, p, dev_ls = test(model, loss, dev_loader, device)
        train_ls = train(model, loss, optimizer, train_loader, epoch + 1, EPOCH, device)

        if f1 > dl['f1']:
            dl['acc'], dl['f1'], dl['r'], dl['p'], dl['loss'] = acc, f1, r, p, dev_ls
            torch.save(model.state_dict(), save_path)
        if dev_ls < bl:
            bl = dev_ls
            count = 0
        else:
            count += 1
        if count > patience:
            tqdm.write('停止训练')
            break
        tqdm.write('epoch:{},acc:{},f1:{},r:{},p:{},loss:{}'.format(epoch, acc, f1, r, p, dev_ls))
        tqdm.write('best->acc:{},f1:{},r:{},p:{}'.format(dl['acc'],
                                                         dl['f1'],
                                                         dl['r'],
                                                         dl['p']))

        x.append(epoch)
        train_loss.append(train_ls)
        dev_loss.append(dev_ls)
        # 动态绘制loss曲线
        plt.plot(x, train_loss, color='red', label='train_loss')
        plt.plot(x, dev_loss, color='blue', label='dev_loss')

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.pause(0.1)

    end_time = time.time()
    run_time = round(end_time - start_time)
    # 计算时分秒
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    # 输出
    print('运行时间：{}时{}分{}秒'.format(hour, minute, second))

    spe={'note': args.sep, 'time': '{}时{}分{}秒'.format(hour, minute, second)}

    model.load_state_dict(torch.load(save_path))
    acc, f1, r, p, test_ls = test(model, loss, test_loader, device)
    if f1 > best['performance']['f1']:
        best['performance']['acc'], best['performance']['f1'], best['performance']['r'], best['performance'][
            'p'] = acc, f1, r, p
        best['config'] = init_log()
        best['spe']= spe
        with open(data_save_path, 'w', encoding='utf-8') as f:
            json.dump(best, f, ensure_ascii=False, indent=4)
        # torch.save(model.state_dict(), save_path)

    # 实验日志
    with open(log_path + '/{}.json'.format(end_time), 'w', encoding='utf-8') as f:
        json.dump({'performance': {'acc': acc, 'f1': f1, 'r': r, 'p': p},
                   'config': init_log(),
                   'spe':spe}, f,
                  ensure_ascii=False,
                  indent=4)

    print('测试集结果为:\nacc:{},f1:{},r:{},p:{},loss:{}'.format(acc, f1, r, p, test_ls))
    print('模型最佳结果为:\nacc:{},f1:{},r:{},p:{}'.format(best['performance']['acc'],
                                                           best['performance']['f1'],
                                                           best['performance']['r'],
                                                           best['performance']['p']))
