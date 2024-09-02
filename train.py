from dataloader.spacedataset import Spacedateset
import os
# from semantic_predict import predict
import functools
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from module.Spacenet import backbone,NLSPNModel,backbone_with_dream,A_CSPN_plus_plus
from argparse import ArgumentParser, Namespace
from typing import List
from torch.utils.data.dataset import random_split
from loss.Loss import MaskMSEloss,MaskL1loss,L1loss,MSEloss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import AverageMeter,Result
import time
from helper import Logger

os.environ['CUDA_VISIBLE_DEVICES']="0,1,2"

def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser(description='Sparse-to-Dense')
    # train parameter
    parser.add_argument('--with_shadow',default=False,type=bool,help='use shadow image training or not')
    parser.add_argument('--n', type=str,default="CSPN",help='model name of the training')
    parser.add_argument('--useplane',default=False,type=bool,help='use plane constraint or not')
    parser.add_argument('--workers',default=8,type=int,metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',default=100,type=int,metavar='N',help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch',default=0,type=int,metavar='N',help='manual epoch number (useful on restarts)')
    parser.add_argument('--start-epoch-bias',default=0,type=int,metavar='N',help='manual epoch number bias(useful on restarts)')
    parser.add_argument('-b',
                        '--batch-size',default=8,type=int,help='mini-batch size (default: 1)')
    parser.add_argument('-c', '--criterion',metavar='LOSS',default='l1',help='criterion (default: l2)')
    parser.add_argument('--lr',
                        '--learning-rate',default=1.2727e-3,type=float,metavar='LR',help='initial learning rate (default 1e-5)')
    parser.add_argument('--weight-decay',
                        '--wd',default=1e-6,type=float,metavar='W',help='weight decay (default: 0)')
    parser.add_argument('--print-freq',
                        '-p',default=50,type=int,metavar='N',help='print frequency (default: 10)')
    parser.add_argument('--resume',default="",type=str,metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint',default="./checkpoints/",type=str,metavar='PATH',help='path to latest checkpoint ')
    parser.add_argument('--pretrain_backbone',default="./checkpoints/backbone_best.pth",help="backbone pretained")
    parser.add_argument('--save_epoch',default=50,type=int)
    parser.add_argument('-pat','--patience',default=4,type=int,help='mini-batch size (default: 1)')
    parser.add_argument('--val',type=str,default=False,help='validation set or not')
    parser.add_argument('--jitter',type=float,default=0.1,help='color jitter for images')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
    parser.add_argument('--s_p',   default="",type=str, metavar='PATH')
    parser.add_argument('--val_results',   default="val_results/",type=str, metavar='PATH')
    parser.add_argument('--test', action="store_true", default=False,
                        help='save result kitti test dataset for submission')
    parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')
    parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                        choices=[1, 2, 4],
                        help='CSPN++ dilation rate')
    parser.add_argument('--train_option',type=str,default='mask')
    parser.add_argument('--mask',type=bool,default=False)
    parser.add_argument('--model_name',type=str,default='CSPN',help='train backbone model or NLSPN model')
    # dataset
    parser.add_argument('--shape',type=tuple,default=(640,960))
    parser.add_argument('--prefill', type=bool, default=True, help='whether to use prefill')
    parser.add_argument('--prop_kernel',type=int,default=5)
    parser.add_argument('--preserve_input',
                    action='store_true',
                    default=False,
                    help='preserve input points by replacement')
    parser.add_argument('--affinity',
                        type=str,
                        default='TGASS',
                        choices=('AS', 'ASS', 'TC', 'TGASS'),
                        help='affinity type (dynamic pos-neg, dynamic pos, '
                            'static pos-neg, static pos, none')
    parser.add_argument('--affinity_gamma',
                        type=float,
                        default=0.5,
                        help='affinity gamma initial multiplier '
                            '(gamma = affinity_gamma * number of neighbors')
    parser.add_argument('--conf_prop',
                        action='store_true',
                        default=True,
                        help='confidence for propagation')
    parser.add_argument('--no_conf',
                        action='store_false',
                        dest='conf_prop',
                        help='no confidence for propagation')
    parser.add_argument('--legacy',
                        action='store_true',
                        default=False,
                        help='legacy code support for pre-trained models')
    parser.add_argument('--from_scratch',
                    action='store_true',
                    default=True,
                    help='train from scratch')
    parser.add_argument('--use_pose', type=bool, default=False, help='whether to use pose')
    parser.add_argument(
    '-m',
    '--train-mode',
    type=str,
    default="Only_fill",
    choices=["backbone","NLSPN"],
    help='dense | sparse | photo | sparse+photo | dense+photo')
    parser.add_argument(
    '--rank-metric',
    type=str,
    default='rmse')
    parser.add_argument('--prop_time',
                    type=int,
                    default=10,
                    help='number of propagation')
    return parser.parse_args()


def get_model():
    args = parse_args()
    if args.model_name == "NLSPN":
        net = NLSPNModel(args)
    elif args.model_name == "CSPN":
        net = A_CSPN_plus_plus(args)
    else:
        net = None
    backbone_net = backbone_with_dream(args)
    return args,net,backbone_net


def save_checkpoint(state, epoch, output_directory,train_type):
    checkpoint_filename = os.path.join(output_directory,
                                       train_type+'_' + str(epoch) + '.pth')
    torch.save(state, checkpoint_filename)


def iterate(iter_mode,args,dataloader,model,optimizer,epoch,device,Loss,logger):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    if iter_mode == 'train':
        model.train()
    else:
        model.eval()
    if epoch <5:
        w_1 ,w_2,w_3 = 0.1,5,0.1
    else:
        w_1, w_2, w_3 = 0,5,0
    val_loss = 0
    for i, data in enumerate(dataloader):
        start = time.time()
        if iter_mode == 'train':
            optimizer.zero_grad()
        rgb = data['rgb'].to(device)
        if args.with_shadow:
            rgb_gt = data['rgb_gt'].to(device)
            RGB_loss = MSEloss()
        sparse = data['sparse'].to(device)
        gt = data['gt'].to(device)
        plane = data['plane'].to(device)
        batch = {'rgb': rgb, 'd': sparse,'plane': plane}
        valid = data['mask'].to(device)
        gt = torch.where(valid>0,gt,torch.zeros_like(gt))
        data_time = time.time() - start
        if iter_mode == 'train':
            if args.model_name == 'NLSPN':
                out = model(batch)
                pred = out['pred']
            elif args.model_name == "CSPN":
                st1_pred, st3_pred, pred = model(batch)
            else:
                st1_pred, rgb_pred, st3_pred, pred = model(batch)
            gpu_time = time.time() - start
            depth_loss = Loss(pred,gt)
            if args.model_name =='NLSPN' or args.model_name == "CSPN":
                loss = depth_loss
            
            else:
                loss_1 = Loss(st1_pred,gt)
                loss_2 = Loss(rgb_pred,rgb)
                if args.with_shadow:
                    loss_2 = RGB_loss(rgb_pred,rgb_gt)
                loss_3 = Loss(st3_pred,gt)
                loss = w_1*loss_1 + w_2*loss_2 + w_3*loss_3+depth_loss
                # loss = w_1*loss_1 + w_3*loss_3+depth_loss

            loss.backward()
            optimizer.step()
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                            'Loss {loss:.4f} '.format(epoch, i, len(dataloader), loss=loss.item()))

        with torch.no_grad():
            if iter_mode == 'val':
                if args.model_name == 'NLSPN':
                    out = model(batch)
                    pred = out['pred']
                elif args.model_name == "CSPN":
                    st1_pred, st3_pred, pred = model(batch)
                else:
                    st1_pred, rgb_pred, st3_pred, pred = model(batch)
                pred = torch.where(valid>0,pred,torch.zeros_like(pred))
                gpu_time = time.time() - start
                depth_loss = Loss(pred,gt)
                val_loss += depth_loss.item()
            result = Result()
            result.evaluate(pred.data, gt.data)
            [
                m.update(result, gpu_time, data_time, n=args.batch_size)
                for m in meters
            ]
            logger.conditional_print(iter_mode, i, epoch, len(dataloader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(iter_mode, i, data, pred,
                                                   epoch)
            logger.conditional_save_pred(iter_mode, i, pred, epoch)
                                     
    avg = logger.conditional_save_info(iter_mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(iter_mode, avg, epoch)
    if is_best and not (iter_mode == "train"):
        logger.save_img_comparison_as_best(iter_mode, epoch)
    logger.conditional_summarize(iter_mode, avg, is_best)

    return is_best,avg


def train():
    args,net,backbone_net = get_model()
    args.result = os.path.join('.', 'Dream')
    logger = Logger(args)
    args.prefill = True
    args.val = False
    train_set = Spacedateset(args)
    args.val = True
    val_set = Spacedateset(args)
    if args.model_name == 'NLSPN' or args.model_name == 'CSPN':
        checkpoint = torch.load(args.pretrain_backbone,map_location=torch.device('cpu'))
        net.backbone.load_state_dict(checkpoint)
        for p in net.backbone.parameters():
            p.requires_grad = False
    else:
        net = backbone_net
        # if args.with_shadow:
        #     checkpoint = torch.load(args.pretrain_backbone,map_location=torch.device('cpu'))
        #     net.load_state_dict(checkpoint)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"
    net = torch.nn.DataParallel(net,device_ids=[0,1,2])
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        net = net.to(device)

    Loss = MaskMSEloss() if args.criterion == 'l2' else MaskL1loss()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model_named_params = [
            p for _, p in net.named_parameters() if p.requires_grad
        ]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, verbose=True)
    for epoch in range(args.start_epoch,args.epochs):
        iterate('train',args,train_loader,net,optimizer,epoch,device,Loss,logger=logger)
        for p in net.parameters():
            p.requires_grad = False
        is_best ,avg= iterate('val',args,val_loader,net,optimizer,epoch,device,Loss,logger=logger)
        if is_best:
            torch.save(net.module.state_dict(), args.checkpoint+args.n+'_'+args.model_name+"_best.pth")
        scheduler.step(avg.rmse)
        for p in net.parameters():
            p.requires_grad = True
        if args.model_name == 'NLSPN' or args.model_name == 'CSPN':
            for p in net.module.backbone.parameters():
                p.requires_grad = False
        if epoch % args.save_epoch == 0:
            save_checkpoint(net.module.state_dict(), epoch, args.checkpoint,args.n+'_'+args.model_name)

if __name__ == '__main__':
    train()