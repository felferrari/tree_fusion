from tqdm import tqdm
import torch
import os
from torchmetrics.classification import MulticlassF1Score
from torch.nn.functional import one_hot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from conf import general

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def train_loop(dataloader, model, loss_fn, optimizer):
    """Executes a train loop epoch

    Args:
        dataloader (Dataloader): Pytorch Dataloader to extract train data
        model (Module): model to be trained
        loss_fn (Module): Loss Criterion
        optimizer (Optimizer): Optimizer to adjust the model's weights

    Returns:
        float: average loss of the epoch
    """
    train_loss, f1_sum, steps = 0, 0, 0
    pbar = tqdm(dataloader)
    f1 = MulticlassF1Score(num_classes=general.N_CLASSES, ignore_index = general.DISCARDED_CLASS)
    
    for (X, y) in pbar:
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        #loss = loss_fn(pred[:, :-1, :, :], one_hot(y, 10).permute((0, 3, 1, 2))[:, :-1, :, :].float())
        f1_sum += f1(pred.to('cpu'), y.to('cpu'))
        steps += 1
        train_loss += loss.item()
        pbar.set_description(f'Train Loss: {train_loss/steps:.4f}, F1-Score:{f1_sum/steps:.4f}')

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = train_loss/steps
    print(f"Average Training Loss: {loss:.4f}, Average F1-Score: {f1_sum/steps:.4f}")
    return loss

def val_loop(dataloader, model, loss_fn):
    """Evaluates a validation loop epoch

    Args:
        dataloader (Dataloader): Pytorch Dataloader to extract validation data
        model (Module): model to be evaluated
        loss_fn (Module): Loss Criterion

    Returns:
        float: average loss of the epoch
    """
    num_steps = len(dataloader)
    val_loss, f1_sum, steps = 0, 0, 0
    f1 = MulticlassF1Score(num_classes=general.N_CLASSES, ignore_index = general.DISCARDED_CLASS)
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for X, y in pbar:
            pred = model(X)
            loss = loss_fn(pred, y)
            steps += 1
            f1_sum += f1(pred.to('cpu'), y.to('cpu'))
            val_loss += loss.item()
            pbar.set_description(f'Val Loss: {val_loss/steps:.4f}, F1-Score:{f1_sum/steps:.4f}  ')

    val_loss /= num_steps
    print(f"Average Validation Loss: {val_loss:.4f}, Average F1-Score: {f1_sum/steps:.4f}")
    return val_loss

def val_sample_image(dataloader, model, path_to_samples, epoch):
    sample = next(iter(dataloader))
    label = sample[1]
    x = sample[0]
    pred = model(x).argmax(axis=1)
    plt.close('all')
    for i, l in enumerate(label):
        figure, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,5))
        p = pred[i]
        cmap = plt.get_cmap('tab20', 10)
        im0 = ax[0].imshow(l.cpu(), cmap = cmap, vmin=-0.5, vmax = 9.5)
        ax[0].title.set_text('Label')
        im1 = ax[1].imshow(p.cpu(), cmap = cmap, vmin=-0.5, vmax = 9.5)
        ax[1].title.set_text(f'Prediction Epoch {epoch+1:03d}')
        
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(im1, cax=cax, orientation='vertical', ticks = np.arange(10))

        figure.savefig(os.path.join(path_to_samples, f'sample_{i}_{epoch}.png'), bbox_inches='tight')
        figure.clf()
        plt.close()

class EarlyStop():
    def __init__(self, train_patience, path_to_save, min_delta = 0, min_epochs = None) -> None:

        self.train_pat = train_patience
        self.no_change_epochs = 0
        self.better_value = None
        self.path_to_save = path_to_save
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.decorred_epochs = 0

    def testEpoch(self, model, val_value):
        self.decorred_epochs+=1
        if self.min_epochs is not None:
            if self.decorred_epochs <= self.min_epochs:
                print(f'Epoch {self.decorred_epochs} from {self.min_epochs} minimum epochs. Validation value:{val_value:.4f}' )
                return False
        if self.better_value is None:
            self.no_change_epochs += 1
            self.better_value = val_value
            print(f'First Validation Value {val_value:.4f}. Saving model in {self.path_to_save}' )
            torch.save(model.state_dict(), self.path_to_save)
            return False
        delta = -(val_value - self.better_value)
        if delta > self.min_delta:
            self.no_change_epochs = 0
            print(f'Validation value improved from {self.better_value:.4f} to {val_value:.4f}. Saving model in {self.path_to_save}' )
            torch.save(model.state_dict(), self.path_to_save)
            self.better_value = val_value
            return False
        else:
            self.no_change_epochs += 1
            print(f'No improvement for {self.no_change_epochs}/{self.train_pat} epoch. Better Validation value is {self.better_value:.4f}' )
            if self.no_change_epochs > self.train_pat:
                return True

