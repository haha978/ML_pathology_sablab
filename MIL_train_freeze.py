import sys
import os
import numpy as np
import argparse
import random
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import h5py

parser = argparse.ArgumentParser(description='Glioma vs non-glioma binary classifcation task')
parser.add_argument('--lib', type=str, default='', help='path to dictionary that contains list of slide numbers for training/validation/testing')
parser.add_argument('--dbase', type=str, default='/home/jm2239/model/slides_dbase.hdf5', help='path to the database that contains preprocessed tiles')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size for inference (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    #cnn
    model = models.resnet34(pretrained = True)
    #freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print("Model parameters (excluding last linear layer) frozen")
    model.fc = nn.Linear(model.fc.in_features, 2)

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.nepochs*0.5, args.nepochs*0.75], gamma = 0.1)
    model.cuda()

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    lib = torch.load(args.lib)
    #slide list contains training
    slide_list = lib['train']
    val_slide_list = lib['validate']

    for epoch in range(args.nepochs):
        scheduler.step()
        #accumulates loss for every slide in slide_list
        total_loss = 0
        preds = []
        reals = []
        random.shuffle(slide_list)
        for idx, slide in enumerate(slide_list):
            train_dset = Slidedataset(slide,args.dbase,trans)
            inf_loader = torch.utils.data.DataLoader(
                train_dset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
            train_dset.setmode(1)
            train_dset.makeinfdata()
            probs = inference(epoch,inf_loader, model)
            pred = pred_target(max(probs))
            #to calculate error rates
            preds.append(pred)
            reals.append(train_dset.target)
            train_loader = torch.utils.data.DataLoader(
                train_dset,
                batch_size = 64, shuffle = False,
                num_workers = args.workers, pin_memory = False)
            #return index
            idxs = argtopk(train_dset.tile_idxs,probs)
            #start from here
            train_dset.maketraindata(idxs)
            train_dset.shuffletraindata()
            train_dset.setmode(2)
            #start train
            loss = train(epoch, train_loader, model, criterion, optimizer)
            total_loss += loss
        train_loss = total_loss/len(slide_list)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, train_loss))
        fconv = open(os.path.join(args.output, 'Loss_train.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,train_loss))
        fconv.close()
        #get_errors
        err,fpr,fnr = calc_err(preds, reals)
        fconv = open(os.path.join(args.output,'Errors_train.csv'),'a')
        fconv.write('{},error,{}\n'.format(epoch+1, err))
        fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
        fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
        fconv.close()
        print('Training\tError: {}\t'.format(err))
        #Validation
        if (epoch+1) % args.test_every == 0:
            val_total_loss = 0
            val_preds = []
            val_reals = []
            for idx, slide in enumerate(val_slide_list):
                val_dset = Slidedataset(slide,args.dbase,trans)
                inf_loader = torch.utils.data.DataLoader(
                    val_dset,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=False)
                val_dset.setmode(1)
                val_dset.makeinfdata()
                probs = inference(epoch,inf_loader,model)
                pred = pred_target(max(probs))
                #accumulate to accuracy
                val_preds.append(pred)
                val_reals.append(val_dset.target)
                train_loader = torch.utils.data.DataLoader(
                    val_dset,
                    batch_size = 64, shuffle = False,
                    num_workers = args.workers, pin_memory = False)
                #return index
                idxs = argtopk(val_dset.tile_idxs,probs)
                val_dset.maketraindata(idxs)
                val_dset.shuffletraindata()
                val_dset.setmode(2)
                loss = get_loss(train_loader,model,criterion)
                val_total_loss += loss
            val_loss = val_total_loss/len(val_slide_list)
            print('Validation\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, val_loss))
            fconv = open(os.path.join(args.output, 'Loss_validate.csv'), 'a')
            fconv.write('{},loss,{}\n'.format(epoch+1,loss))
            fconv.close()
            err,fpr,fnr = calc_err(val_preds, val_reals)
            fconv = open(os.path.join(args.output,'Errors_validate.csv'),'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            print('Validation\tError: {}\t'.format(err))
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_best_{}_{}.pth'.format(epoch, best_acc)))

def pred_target(prob):
    if prob >= 0.5:
        return 1
    else:
        return 0

def get_loss(loader,model,criterion):
    model.eval()
    running_loss = 0.
    with torch.no_grad():
        for i, (input,target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    print("pred:" + str(pred))
    print("real:" + str(real))
    print("neq is " + str(neq))
    print("pred = 1:" + str(np.logical_and(pred==1, neq).sum()) + " real = 0:" + str((real == 0).sum()))
    print("pred = 0:" + str(np.logical_and(pred==0, neq).sum()) + " real = 1:" + str((real == 1).sum()))
    return err, fpr, fnr

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def argtopk(tile_idxs, probs ,k=64):
    """
    Returns list of 64 tiles' indexs that have the highest probability of being
    positive
    """
    #sort
    tile_idx_prob_l= sorted(list(zip(tile_idxs,probs)),key= lambda x : x[1])
    max_idx_prob_list= tile_idx_prob_l[-64:]
    idxs = []
    for idx_prob in max_idx_prob_list:
        idxs.append(idx_prob[0])
    return idxs

def inference(run, loader, model):
    model.eval()
    #vector with length of the dataset
    probs = torch.FloatTensor(len(loader.dataset))
    #dont want to backpropaate and save gradients when evaluating
    with torch.no_grad():
        #input is an element of loader
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            #dim=1 each element in the row will be changed btw [0,1] model(input) has [a, b] where a, b [0,1]
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    #returns probabilities that contains probability that the each patch is positive
    return probs.cpu().numpy()


class Slidedataset(data.Dataset):
    def __init__(self, slide_num , dbase_path, transform=None):
        self.dbase_path = dbase_path
        dbase = h5py.File(dbase_path,'r')
        self.slide_num = slide_num
        self.tile_num = len(dbase[slide_num]['tiles'])
        self.transform = transform
        self.mode = None
        self.target = int(dbase[slide_num]['targets'][()])
        dbase.close()
    def makeinfdata(self):
        self.tile_idxs = random.sample(range(self.tile_num),self.tile_num)
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.tile_idxs[x],self.target) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        #for inference
        if self.mode == 1:
            dbase = h5py.File(self.dbase_path,'r')
            tileIDX = self.tile_idxs[index]
            img = dbase[self.slide_num]['tiles'][tileIDX]
            dbase.close()
            if self.transform is not None:
                img = self.transform(img)
            return img
        #for training
        elif self.mode == 2:
            dbase = h5py.File(self.dbase_path,'r')
            tileIDX, target = self.t_data[index]
            img = dbase[self.slide_num]['tiles'][tileIDX]
            dbase.close()
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.tile_idxs)
        elif self.mode == 2:
            return len(self.t_data)

if __name__ == '__main__':
    main()
