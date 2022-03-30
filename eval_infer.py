'''
pip install scikit-learn
pip install seaborn
pip install pandas
'''

import os
import time, datetime
import numpy as np
import torch
import torchvision.transforms as transforms
import helpers
import argparse
from config.utils import *
import lp.db_semisuper as db_semisuper
import lp.db_eval as db_eval
from models import *
import torch.backends.cudnn as cudnn
# 지표 추가
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.cuda import amp

def create_model(num_classes,args):
    model_choice = args.model
    
    if model_choice == "resnet18":
        model = resnet18(num_classes)
        
    elif model_choice == "resnet50":
        model = resnet50(num_classes)
        
    elif model_choice == "wrn-28-2":
        model = build_wideresnet(28,2,0,num_classes)
    
    elif model_choice == "wrn-28-8":
        model = build_wideresnet(28,8,0,num_classes)
        
    elif model_choice == "cifarcnn":
        model = cifar_cnn(num_classes)

    model.to(args.device)
    cudnn.benchmark = True    
    return model



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# def evaluate(eval_loader, model, args, num_classes = 4):
def evaluate(eval_loader, model, args, num_classes):
    meters = AverageMeterSet()    
    con_mat = np.zeros((args.num_classes, args.num_classes))
    class_score = []
    # 성능지표 추가
    y_true = []
    y_pred = []
    # maxk = 5
    maxk = 1
    if num_classes < 5:
        maxk = num_classes
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            batch_size = targets.size(0)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            # outputs,_ = model(inputs)
            with amp.autocast(enabled=True):
                outputs,_ = model(inputs)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, maxk))
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, maxk))
            meters.update('top1', prec1.item(), batch_size)
            meters.update('error1', 100.0 - prec1.item(), batch_size)
            meters.update('top5', prec5.item(), batch_size)
            meters.update('error5', 100.0 - prec5.item(), batch_size)
             # f1 score, accuracy, precision, recall
            _, preds = torch.max(outputs, 1)    # 추론이미지별 acc 가장 높은 클래스인덱스
            _, indices = torch.sort(outputs, descending=True)   # 추론이미지별 acc 가장 높은 순서대로 모든 클래스인덱스 나열
            for i, (t,p) in enumerate(zip(targets.view(-1), preds.view(-1))):
                # 가로 : predicted label, 세로 : GT label
                con_mat[t.long(), p.long()] += 1
                percent = torch.nn.functional.softmax(outputs, dim=1)[i] * 100
                # GT idx & pred idx & score desc
                tp_score = [(t.item(), idx.item(), percent[idx].item()) for idx in indices[i,]] # class_names[idx]
                class_score.append(tp_score)
            y_true.append(targets.cpu().data.tolist())
            y_pred.append(preds.cpu().tolist())
            # f1 += f1_score(targets.cpu().data, preds.cpu(), average=None, zero_division=0)
            # accu += accuracy_score(targets.cpu().data, preds.cpu())
            # precision += precision_score(targets.cpu().data, preds.cpu(), average=None, zero_division=0)
            # recall += recall_score(targets.cpu().data, preds.cpu(), average=None, zero_division=0)
        # f1 score, accuracy, precision, recalls
        # avg_f1 = f1 / len(eval_loader) 
        # avg_acc = accu / len(eval_loader) 
        # avg_precision = precision / len(eval_loader)
        # avg_recall = recall / len(eval_loader)
        class_f1_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
        class_pre_arr = precision_score(y_true, y_pred, average=None, zero_division=0)
        class_rec_arr = recall_score(y_true, y_pred, average=None, zero_division=0)
        class_acc_arr = accuracy_score(y_true, y_pred)
        print('-' * 10)
        print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
              .format(top1=meters['top1'], top5=meters['top5']))
        avg_acc = np.mean(class_acc_arr)
        avg_f1 = np.mean(class_f1_arr)
        avg_pre = np.mean(class_pre_arr)
        avg_rec = np.mean(class_rec_arr)
        print('-' * 10)
        print("Accuracy : {:.4f}".format(avg_acc))
        print("F1-score : {:.4f}".format(avg_f1))
        print("Precision : {:.4f}".format(avg_pre))
        print("Recall : {:.4f}".format(avg_rec))
        print('-' * 10)
        make_cm(eval_loader, con_mat, args)
    return meters['top1'].avg, meters['top5'].avg


def make_cm(eval_loader, con_mat, args):
    labels = eval_loader.__dict__['dataset'].__dict__['targets']
    labels = torch.tensor(labels)
    labels = labels.to(args.device)
    # class_name 함수 호출
    class_name = args.class_list
    if "D4" in class_name:
        class_name[class_name.index("D4")] = "D3"
    elif "S4" in class_name:
        class_name[class_name.index("S4")] = "S3"
    elif "R1" in class_name:
        class_name[class_name.index("R1")] = "N1"
        class_name[class_name.index("R2")] = "N2"
    # elif "R2" in class_name:
    #     class_name[class_name.index("R2")] = "N2"
    print(class_name)
    cmt = torch.zeros(len(class_name), len(class_name), dtype=torch.int64)
    # confusion matrix to DF
    df_cm = pd.DataFrame(con_mat, index=class_name, columns=class_name).astype(int)
    plt.ioff()
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 10})
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    heatmap.set_title('confusion matrix', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('Ground Truth Label', fontsize=10)
    plt.savefig(os.path.join(args.cm_save, f'laplace_{args.dataset}_cm.png'), dpi=300)
    return


def get_eval_transform():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return eval_transformation

def get_eval_loader(eval_transformation, evaldir, args):

    eval_dataset = db_eval.DBE(evaldir, False, eval_transformation)
    args.class_list = eval_dataset.classes
    print(args.class_list)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)
        
    return eval_loader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='S_trans', help='dataset name')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--num_classes', type=int, default=4, help='classes')
    parser.add_argument('--weight', type=str, default='./result/S_trans/20220228_183337/resnet18_115_prec93.0_acc0.8863636363636364_best.pth', help='model name')
    parser.add_argument('--num_labeled', type=int, default=50, help='labeled number')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--cm_save', type=str, default='./result/S_trans/20220228_183337')
    parser.add_argument('--best_weight', type=str, default='./result/S_trans/20220228_183337/resnet18_115_prec93.0_acc0.8863636363636364_best.pth', help='model name')
    args = parser.parse_args()
    args.workers = 4 * torch.cuda.device_count()
    
    # 모델
    # best_weight='./result/S_trans/20220228_183337/resnet18_115_prec93.0_acc0.8863636363636364_best.pth'
    # best_weight='./result/R_trans/20220302_094148/resnet18_140_prec72.0_acc0.678030303030303.pth'
    best_weight=args.best_weight
    #### Create Model
    args.device = torch.device('cuda')
    model = create_model(args.num_classes, args)
    model.load_state_dict(torch.load(best_weight))

    # 데이터로더
    eval_transformation=get_eval_transform()
    evaldir = f'data-local/images/custom/{args.dataset}/test'
    eval_loader = get_eval_loader(eval_transformation, evaldir, args)


    # evaluate(eval_loader, model, args, num_classes = 4)
    evaluate(eval_loader, model, args, num_classes = args.num_classes)