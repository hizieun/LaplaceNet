import os
import time, datetime
import numpy as np
import torch
from config import datasets, cli
import math
import argparse

def print_path():
    try:
        python_path=os.environ['PYTHONPATH']
        split_list=python_path.split(';')	
        split_list.sort()	
        python_path_str='\n'.join(split_list)
        print(f'python_path={python_path_str}')
    except Exception as e:
        print(e)
        
    os_path=os.environ['PATH']
    split_list=os_path.split(';')
    split_list.sort()	
    os_path_str='\n'.join(split_list)
    print(f'os_path={os_path_str}')

# 설치된 dll을 찾지못하는 경우 발생 
# 강제로 path에 추가 
def set_path_4_faiss():	
    os.environ['PATH']+=';C:/Users/USER/miniconda3/envs/laplacenet_env/Library/bin;'
    os.environ['PATH']+='C:/Users/USER/.conda/envs/laplacenet_env/bin;'
	
# set_path_4_faiss()
import helpers

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--model', type=str, default='wrn-28-8', help='model name')
parser.add_argument('--num-labeled', type=int, default=4000, help='labeled number')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--labeled-batch-size', type=int, default=10, help='labeled batch size')
parser.add_argument('--batch-size', type=int, default=100, help='batch size')
parser.add_argument('--aug-num', type=int, default=3, help='augmentaion number')
parser.add_argument('--label-split', type=int, default=12, help='label split ')
# TODO: label split의 의미는?  코드에서 찾아볼 것 -> data label txt 파일의 이름
parser.add_argument('--progress', action="store_true", default=True, help='progess bar')


parser.add_argument('--eval-subdir', type=str, default='test', help='evalutation subdir')
parser.add_argument('--knn', type=int, default=50, help='knn')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--nesterov', action="store_true", default=True, help='nesterov ')
# parser.add_argument('--num-steps', type=int, default=250000,help='num-steps')
parser.add_argument('--num-steps', type=int, default=2500,help='num-steps')
parser.add_argument('--train-subdir', type=str, default='train+val', help='train subdir')
parser.add_argument('--weight-decay', type=float, default=0.0005, help='momentum')
parser.add_argument('--datadir', type=str, default='', help='custom dataset sub directory')
parser.add_argument('--num-classes', type=int, default=4, help='if you used custom dataset, set num classes for custom/datadir')

def laplacenet():
    #### Get the command line arguments
    args = parser.parse_args()
    # args = cli.parse_commandline_args()

    print(f'args={args}')
    args = helpers.load_args(args)
    # print(f'args={args}')
    now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.file = args.model + "_" + args.dataset + "_" + args.datadir + "_" + str(args.num_labeled) + "_" + str(args.label_split) + "_" + str(args.num_steps) + "_" + str(args.aug_num) + "_" +now+ ".txt"
    # resnet18_custom_R_trans_50_12_2500_3_20220302_094148.txt
    #### Save model and logs dir
    save_dir = os.path.join('./result', args.datadir, now)
    os.makedirs(save_dir, exist_ok=True)

    args.file = os.path.join(save_dir, args.file)
    print(f'args.file={args.file}')

    #### Load the dataset
    dataset_config = datasets.__dict__[args.dataset]()
    if args.dataset =='custom':
        dataset_config['num_classes'] = args.num_classes
        dataset_config['datadir'] = os.path.join(dataset_config['datadir'], args.datadir)
    num_classes = dataset_config.pop('num_classes')
    args.num_classes = num_classes
    


    #### Create loaders
    #### train_loader loads the labeled data , eval loader is for evaluation
    #### train_loader_noshuff extracts features 
    #### train_loader_l, train_loader_u together create composite batches
    #### dataset is the custom dataset class
    train_loader, eval_loader , train_loader_noshuff , train_loader_l , train_loader_u , dataset = helpers.create_data_loaders_simple(**dataset_config, args=args)

    #### Create Model and Optimiser 
    args.device = torch.device('cuda')
    model = helpers.create_model(num_classes,args)
    # print(model)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,weight_decay=args.weight_decay, nesterov=args.nesterov)

    #### Transform steps into epochs
    num_steps = args.num_steps
    ini_steps = math.floor(args.num_labeled/args.batch_size)*100
    ssl_steps = math.floor( len(dataset.unlabeled_idx) / ( args.batch_size - args.labeled_batch_size))
    args.epochs = 10 + math.floor((num_steps - ini_steps) / ssl_steps)
    args.lr_rampdown_epochs = args.epochs + 10


    #### Information store in epoch results and then saved to file
    global_step = 0
    epoch_results = np.zeros((args.epochs,6))

    best_acc = 0.75
    EPOCH_CRITERIA=10
    # EPOCH_CRITERIA=2
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        #### Extract features and run label prop on graph laplacian
        if epoch >= EPOCH_CRITERIA:
            dataset.feat_mode = True
            feats = helpers.extract_features_simp(train_loader_noshuff,model,args)  
            dataset.feat_mode = False          
            dataset.one_iter_true(feats,k = args.knn, max_iter = 30, l2 = True , index="ip") 

        #### Supervised Initilisation vs Semi-supervised main loop
        start_train_time = time.time()  
        if epoch < EPOCH_CRITERIA:
            print("Supervised Initilisation:", (epoch+1), "/" , EPOCH_CRITERIA )
            for i in range(EPOCH_CRITERIA):
                print(f'initialize i={i}')
                print(time.time())
                global_step = helpers.train_sup(train_loader, model, optimizer, epoch, global_step, args)                     
        if epoch >= EPOCH_CRITERIA:
            global_step = helpers.train_semi(train_loader_l, train_loader_u, model, optimizer, epoch, global_step, args)  

        end_train_time = time.time()
        print("Evaluating the primary model:", end=" ")
        prec1, prec5 = helpers.validate(eval_loader, model, args, global_step, epoch + 1, num_classes = args.num_classes)

        epoch_results[epoch,0] = epoch
        epoch_results[epoch,1] = prec1
        epoch_results[epoch,2] = prec5 
        epoch_results[epoch,3] = dataset.acc
        epoch_results[epoch,4] = time.time() - start_epoch_time
        epoch_results[epoch,5] = end_train_time - start_train_time
        
        # print(args.file)
        np.savetxt(args.file,epoch_results,delimiter=',',fmt='%1.3f')

        if best_acc <= dataset.acc:
            torch.save(model.module.state_dict(), f'{save_dir}/{args.model}_{epoch}_prec{prec1}_acc{dataset.acc}_best.pth')
            best_acc = dataset.acc
            print('saved best weight ... ')
        elif ( prec1 >= 75.0 and dataset.acc >= 0.8 ) or epoch%10 == 0:
            torch.save(model.module.state_dict(), f'{save_dir}/{args.model}_{epoch}_prec{prec1}_acc{dataset.acc}.pth')
            print(f'saved weight epoch={epoch}, prec1={prec1}, acc={dataset.acc} ... ')
# def test_savetext():
# 	a_list=[[  0. ,         31.4,         85.61    ,     0.     ,    108.53928232,   68.06500578],
# 	 [  0.,           0.,           0.,           0.,           0.,    0.        ],
# 	 [  0.,           0.,           0.,           0. ,          0.,    0.        ],
# 	 [  0.,           0.,           0.,           0.  ,         0.,    0.        ]]
# 	a_arr=np.array(a_list)
# 	np.savetxt("a.out",a_arr,delimiter=',',fmt='%1.3f')
#


if __name__ == '__main__':
    laplacenet()
  # test_savetext()


