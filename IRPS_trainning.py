import os
import torch
import argparse
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
from model.model_ImmuSlide import ImmuSlide
from utils.ROC_plot import TwoClassROC
from dataset.ImmuDataset import ImmuDataset

from utils.utils import get_optim, print_network,calculate_error, get_loader, get_simple_loader

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count
def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    cls_train_error = 0.
    cls_train_loss = 0.
    batch_loss = []
    print('\n')
    for batch_idx, (data, label, site, sex) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        sex = sex.float().to(device)

        results_dict = model(data, sex)
        logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']


        cls_logger.log(Y_hat, label)

        cls_loss = loss_fn(logits, label)
        loss = cls_loss
        cls_loss_value = cls_loss.item()
        batch_loss.append(cls_loss_value)

        cls_train_loss += cls_loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, cls loss: {:.4f}, '.format(batch_idx, cls_loss_value ) +
                  'label: {}, sex: {}, age: {}'.format(label.item(), site.item(), sex.item(),
                                                                      data.size(0)))

        cls_error = calculate_error(Y_hat, label)
        cls_train_error += cls_error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    cls_train_loss /= len(loader)
    cls_train_error /= len(loader)

    print('Epoch: {}, cls train_loss: {:.4f}, cls train_error: {:.4f}'.format(epoch, cls_train_loss, cls_train_error))
    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        print('class {}: tpr {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_tpr'.format(i), acc, epoch)


    if writer:
        writer.add_scalar('train/cls_loss', cls_train_loss, epoch)
        writer.add_scalar('train/cls_error', cls_train_error, epoch)
    return (batch_loss,cls_train_loss)



def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    cls_val_error = 0.
    cls_val_loss = 0.

    cls_probs = np.zeros((len(loader), n_classes))
    cls_labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, sex) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)
            sex = sex.float().to(device)

            results_dict = model(data, sex)
            logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

            del results_dict

            cls_logger.log(Y_hat, label)

            cls_loss = loss_fn(logits, label)

            loss = cls_loss
            cls_loss_value = cls_loss.item()

            cls_probs[batch_idx] = Y_prob.cpu().numpy()
            cls_labels[batch_idx] = label.item()

            cls_val_loss += cls_loss_value

            cls_error = calculate_error(Y_hat, label)
            cls_val_error += cls_error


    cls_val_error /= len(loader)
    cls_val_loss /= len(loader)

    if n_classes == 2:
        cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
        cls_aucs = []
        TwoClassROC(cls_labels,cls_probs[:, 1])
    else:
        cls_aucs = []
        binary_labels = label_binarize(cls_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in cls_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], cls_probs[:, class_idx])
                cls_aucs.append(calc_auc(fpr, tpr))
            else:
                cls_aucs.append(float('nan'))

        cls_auc = np.nanmean(np.array(cls_aucs))


    if writer:
        writer.add_scalar('val/cls_loss', cls_val_loss, epoch)
        writer.add_scalar('val/cls_auc', cls_auc, epoch)
        writer.add_scalar('val/cls_error', cls_val_error, epoch)

    print('\nVal Set, cls val_loss: {:.4f}, cls val_error: {:.4f}, cls auc: {:.4f}'.format(cls_val_loss, cls_val_error,
                                                                                 cls_auc))

    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        print('class {}: tpr {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_tpr'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, cls_val_loss, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def train(datasets, cur, args):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets

    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    loss_fn = nn.CrossEntropyLoss()

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    model = ImmuSlide(**model_dict)

    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_loader(train_split, training=True, weighted=args.weighted_sample)
    # val_loader = get_simple_loader(val_split)
    # test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)

    else:
        early_stopping = None
    print('Done!')
    epoch_loss={}
    for epoch in range(args.max_epochs):
        res=train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        epoch_loss[epoch]=res
        # stop = validate(cur, epoch, model, val_loader, args.n_classes,
        #                 early_stopping, writer, loss_fn, args.results_dir)
        #
        # if stop:
        #     break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    writer.close()
    return epoch_loss


def main():
    parser = argparse.ArgumentParser(description='Predicting immunotherapy response based on WSI')
    parser.add_argument('--data_csv', type=str, help='data file')
    parser.add_argument('--n_classes', type=int, default=2,help='number of classes')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_Dict = {'gender':{'female':0,'male':1},
                  'therapy':{'Responder':1,'Nonresponder':0}}
    #trainingDatset = ImmuDataset(csv_file='/Data/yangml/Project/TCGA/Public7_TCGA_Part2/lung/data/LUSC/late_stage_lusc_slide/SVS/Trainning/samples_train_dat_latest.csv',label_Dict=label_Dict)
    trainingDatset = ImmuDataset(
        csv_file= args.data_csv,
        label_Dict=label_Dict)

    validatingDatset = ImmuDataset(csv_file='/Data/yangml/Project/TCGA/Public7_TCGA_Part2/lung/data/LUSC/late_stage_lusc_slide/SVS/Trainning/samples_train_dat_latest.csv',label_Dict=label_Dict)
    datasets=(trainingDatset,validatingDatset)
    out=train(datasets,1,args)
    json_str = json.dumps(out)
    with open('loss_data.json', 'w') as json_file:
        json_file.write(json_str)

if __name__ == "__main__":
    main()
