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
#from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--lib', type=str, default='', help='path to dictionary that contains list of slide numbers for training/validation/testing')
parser.add_argument('--dbase', type=str, default='/home/jm2239/model/slides_dbase.hdf5', help='path to the database that contains preprocessed tiles')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size for inference (default: 128)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    #cnn
    model = models.resnet34(pretrained = False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.cuda()

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    #lib contains train/validate/test dsets
    lib = torch.load(args.lib)
    train_dset = MILdataset(lib['train'], args.dbase, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    val_dset = MILdataset(lib['validate'], args.dbase, trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()
    #loop throuh epochs
    # create summary writer for tensorboard
    #writer = SummaryWriter()
    for epoch in range(args.nepochs):
        train_dset.makeinfdata()
        train_dset.setmode(1)
        #goes into loop for one epoch in inference(epoch, train_loader, model)
        probs = inference(epoch, train_loader, model)
        maxs = group_max(np.array(train_dset.slide_idxs), probs, len(train_dset.targets))
        pred = [1 if x >= 0.5 else 0 for x in maxs]
        err,fpr,fnr = calc_err(pred, train_dset.targets)
        print('Training\tError: {}\t'.format(err))
        # end getting error
        topk = group_argtopk(np.array(train_dset.slide_idxs), probs, args.k)
        #topks are indexs from the max pooling
        train_dset.maketraindata(topk)
        #train_loader is changed and only contains data that are max-pooled
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        #length of the train_loader is changed to self.t_data
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence_new.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()
        #writer.add_scalar('Loss/train',loss,epoch)
        #wrtier.add_scalar('Accuracy/train',(1-err),epoch)
        #wrtier.add_scalar('False-positive rate/train',fpr,epoch)
        #wrtier.add_scalar('False-negative rate/train',fpr,epoch)
        #Validation
        if (epoch+1) % args.test_every == 0:
            val_dset.makeinfdata()
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.slide_idxs), probs, len(val_dset.targets))
            #maxs is used only for the prediction and to calcuate error
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence_new.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #writer.add_scalar('Loss/validate',loss,epoch)
            #wrtier.add_scalar('Accuracy/validate',(1-err),epoch)
            #wrtier.add_scalar('False-positive rate/validate',fpr,epoch)
            #wrtier.add_scalar('False-negative rate/validate',fpr,epoch)
            #Save best model
            #err = (fpr+fnr)/2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_best_{}_{}.pth'.format(epoch, best_acc)))
            #writer.close()

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

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

"""
groups = np.array(train_dset.slideIDX)
data = probs
nmax = len(train_set.targets)
"""
def group_max(groups, data, nmax):
    out = np.empty(nmax)
    #set all elements of numpy array with length nmax to (nan = not an integer)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    #construct an index which marks borders between groups
    index = np.empty(len(groups), 'bool')
    #all elemnts of index are True at this point of code
    index[-1] = True
    #element-wise != operation
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out
    #max pooling for each group
def custom_collate(batch, dataset=None, inf_batch_size = 1):
    if dataset.mode == 1:
        return [batch[i] for i in range(inf_batch_size)]
    elif dataset.mode == 2:
        return [batch[i] for i in range()]


class MILdataset(data.Dataset):
    def __init__(self, slide_list= [] , dbase_path = '', transform=None):
        self.dbase_path = dbase_path
        self.dbase = h5py.File(dbase_path,'r')
        #from dictionary
        self.slides = slide_list
        targets = []
        tile_num_list = []
        for slide_num in self.slides:
            target = self.dbase[slide_num]['targets'][()]
            targets.append(target)
            tile_num = len(self.dbase[slide_num]['tiles'])
            tile_num_list.append(tile_num)
        self.targets = targets
        self.tile_num_list = tile_num_list
        #need available tile numbers for each slide to pull 300 tiles randomly
        self.transform = transform
        self.mode = None
    def setmode(self,mode):
        self.mode = mode
    def makeinfdata(self):
        tile_idxs = []
        slide_idxs = []
        #randomly choose tile idxs from each slide
        #choose all if there are no 300 tiles available
        for i, tile_num in enumerate(self.tile_num_list):
            if tile_num < 300:
                idxs = random.sample(range(tile_num),tile_num)
            else:
                idxs = random.sample(range(tile_num),300)
            tile_idxs.extend(idxs)
            slide_idxs.extend([i]*len(idxs))
        self.tile_idxs = tile_idxs
        self.slide_idxs = slide_idxs
    def maketraindata(self, idxs):
        self.t_data = [(self.slide_idxs[x],self.tile_idxs[x],self.targets[self.slide_idxs[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        #for inference
        if self.mode == 1:
            dbase = h5py.File(self.dbase_path,'r')
            slideIDX = self.slide_idxs[index]
            tileIDX = self.tile_idxs[index]
            img = dbase[self.slides[slideIDX]]['tiles'][tileIDX]
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            dbase = h5py.File(self.dbase_path,'r')
            slideIDX, tileIDX, target = self.t_data[index]
            img = dbase[self.slides[slideIDX]]['tiles'][tileIDX]
            if self.transform is not None:
                img = self.transform(img)
            return img, int(target)
    def __len__(self):
        if self.mode == 1:
            return len(self.tile_idxs)
        elif self.mode == 2:
            return len(self.t_data)

#run the main() in the terminal when 'MIL_train.py' is called from terminal
if __name__ == '__main__':
    main()
