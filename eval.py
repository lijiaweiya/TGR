import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def train(model, criterion, optimizer, train_loader, Epoch, EPOCHS, device):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    running_loss = 0.0
    pred, y = [], []
    for step, data in loop:
        data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.cpu().detach().numpy().item()
        # 累加识别正确的样本数
        pred += out.argmax(dim=1).cpu().numpy().tolist()
        y += data.y.cpu().numpy().tolist()

        # 更新信息
        loop.set_description(f'Epoch [{Epoch}/{EPOCHS}]')
        loop.set_postfix(loss=running_loss / (step + 1),
                         acc=accuracy_score(y_true=y, y_pred=pred),
                         f1=f1_score(y_true=y, y_pred=pred, average='macro'),
                         recall=recall_score(y_true=y, y_pred=pred, average='macro'),
                         precision=precision_score(y_true=y, y_pred=pred, average='macro')
                         )
    return running_loss / len(train_loader)


def test(model,criterion, test_loader, device):
    model.eval()
    pred, y = [], []
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:  # 批遍历测试集数据集。
            data.to(device)
            #print(data.x.shape, data.edge_index.shape, data.batch.shape)
            out = model(data.x, data.edge_index, data.batch)  # 一次前向传播
            loss = criterion(out, data.y)
            running_loss += loss.cpu().detach().numpy().item()
            pred += out.argmax(dim=1).cpu().numpy().tolist()  # 使用概率最高的类别
            y += data.y.cpu().numpy().tolist()  # 检查真实标签
    return (accuracy_score(y_true=y, y_pred=pred),
            f1_score(y_true=y, y_pred=pred, average='macro'),
            recall_score( y_true=y, y_pred=pred, average='macro'),
            precision_score(y_true=y, y_pred=pred, average='macro'),
            running_loss / len(test_loader)
            )
