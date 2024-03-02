import argparse
import os
import logging
import time
import pickle
import torch
import torch.nn as nn
import gc

from datetime import datetime

time_str = str(datetime.now().strftime('%y%m%d%H%M'))

from model.datasets import FastSynergyDataset, FastTensorDataLoader
from model.models import MLP
from model.utils import save_args, save_best_model, find_best_model, arg_min, random_split_indices, calc_stat, conf_inv
from const import SYNERGY_FILE, CELL_FEAT_FILE, CELL2ID_FILE, OUTPUT_DIR, DRUGNAME_2_DRUGBANKID_FILE, DTI_FEAT_FILE, FP_FEAT_FILE, DRUG2ID_FILE

 
def eval_model(model, optimizer, loss_func, train_data, test_data,
               batch_size, n_epoch, patience, gpu_id, mdl_dir):
    tr_indices, es_indices = random_split_indices(len(train_data), test_rate=0.1)
    train_loader = FastTensorDataLoader(*train_data.tensor_samples(tr_indices), batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(*train_data.tensor_samples(es_indices), batch_size=len(es_indices) // 4)
    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data) // 4)
    train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, gpu_id,
                sl=True, mdl_dir=mdl_dir)
    test_loss = eval_epoch(model, test_loader, loss_func, gpu_id)
    test_loss /= len(test_data)
    test_accuracy, test_recall = eval_acc(model, test_loader, loss_func, gpu_id, len(test_data))
    return test_loss, test_accuracy, test_recall


def step_batch(model, batch, loss_func, gpu_id=None, train=True):
    drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat, y_true = batch
    if gpu_id is not None:
        drug1_idx = drug1_idx.cuda(gpu_id)
        drug2_idx = drug2_idx.cuda(gpu_id)
        drug1_fp = drug1_fp.cuda(gpu_id)
        drug2_fp = drug2_fp.cuda(gpu_id)
        drug1_dti = drug1_dti.cuda(gpu_id)
        drug2_dti = drug2_dti.cuda(gpu_id)
        cell_feat = cell_feat.cuda(gpu_id)
        y_true = y_true.cuda(gpu_id)


    if train:
        y_pred = model(drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat)
    else:
        yp1 = model(drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat)
        yp2 = model(drug1_idx, drug2_idx, drug2_fp, drug1_fp, drug2_dti, drug1_dti, cell_feat)
        y_pred = (yp1 + yp2) / 2
    loss = loss_func(y_pred, y_true)
    return loss


def train_epoch(model, loader, loss_func, optimizer, gpu_id=None):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = step_batch(model, batch, loss_func, gpu_id)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def eval_epoch(model, loader, loss_func, gpu_id=None):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch in loader:
            loss = step_batch(model, batch, loss_func, gpu_id, train=False)
            epoch_loss += loss.item()
    return epoch_loss

def step_acc(model, batch, loss_func, gpu_id=None, train=True):
    drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat, y_true = batch
    if gpu_id is not None:
        drug1_idx = drug1_idx.cuda(gpu_id)
        drug2_idx = drug2_idx.cuda(gpu_id)
        drug1_fp = drug1_fp.cuda(gpu_id)
        drug2_fp = drug2_fp.cuda(gpu_id)
        drug1_dti = drug1_dti.cuda(gpu_id)
        drug2_dti = drug2_dti.cuda(gpu_id)
        cell_feat = cell_feat.cuda(gpu_id)
        y_true = y_true.cuda(gpu_id)


    if train:
        y_pred = model(drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat)
    else:
        yp1 = model(drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat)
        yp2 = model(drug1_idx, drug2_idx, drug2_fp, drug1_fp, drug2_dti, drug1_dti, cell_feat)
        y_pred = (yp1 + yp2) / 2
    TP = 0
    TN = 0
    FN = 0
    loss = loss_func(y_pred, y_true)
    for i in range(len(y_pred)):
        if y_true[i] * y_pred[i] > 0:
            if y_true[i] > 0:
                TP += 1
            else:
                TN += 1
        elif y_true[i] > 0:
            FN += 1
    return TP, TN, FN

def eval_acc(model, loader, loss_func, gpu_id, test_len):
    model.eval()
    with torch.no_grad():
        TP = 0
        TN = 0
        FN = 0
        for batch in loader:
            t1, t2, t3 = step_acc(model, batch, loss_func, gpu_id, train=False)
            TP += t1
            TN += t2
            FN += t3
        acc = (TP + TN) / test_len
        recall = TP / (TP + FN)
    return acc, recall


def train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, gpu_id,
                sl=False, mdl_dir=None):
    min_loss = float('inf')
    angry = 0
    for epoch in range(1, n_epoch + 1):
        trn_loss = train_epoch(model, train_loader, loss_func, optimizer, gpu_id)
        trn_loss /= train_loader.dataset_len
        val_loss = eval_epoch(model, valid_loader, loss_func, gpu_id)
        val_loss /= valid_loader.dataset_len
        print("loss epoch " + str(epoch) + ": " + str(trn_loss))
        if val_loss < min_loss:
            angry = 0
            min_loss = val_loss
            if sl:
                # TODO: save best model in case of need
                save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)
                pass
        else:
            angry += 1
            if angry >= patience:
                break
    if sl:
        model.load_state_dict(torch.load(find_best_model(mdl_dir)))
    return min_loss


def create_model(data, hidden_size, gpu_id=None):
    # TODO: use our own MLP model
    # get 256
    model = MLP(data.cell_feat_len() + 2 * 40 + 2 * 256 + 2 * 64, hidden_size, gpu_id)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model


def cv(args, out_dir):

    gc.collect()

    save_args(args, os.path.join(out_dir, 'args.json'))
    test_loss_file = os.path.join(out_dir, 'test_loss.pkl')
    print("cuda available: " + str(torch.cuda.is_available()))

    if torch.cuda.is_available() and (args.gpu is not None):
        gpu_id = args.gpu
    else:
        gpu_id = None


    n_folds = 5
    n_delimiter = 60
    loss_func = nn.MSELoss(reduction='sum')
    test_losses = []
    
    for test_fold in range(n_folds):
        outer_trn_folds = [x for x in range(n_folds) if x != test_fold]
        logging.info("Outer: train folds {}, test folds {}".format(outer_trn_folds, test_fold))
        logging.info("-" * n_delimiter)
        param = []
        losses = []
        for hs in args.hidden:
            for lr in args.lr:
                param.append((hs, lr))
                logging.info("Hidden size: {} | Learning rate: {}".format(hs, lr))
                ret_vals = []
                for valid_fold in outer_trn_folds:
                    inner_trn_folds = [x for x in outer_trn_folds if x != valid_fold]
                    valid_folds = [valid_fold]
                    train_data = FastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, CELL_FEAT_FILE,
                                                    SYNERGY_FILE, DTI_FEAT_FILE, FP_FEAT_FILE, use_folds=inner_trn_folds)
                    valid_data = FastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, CELL_FEAT_FILE,
                                                    SYNERGY_FILE, DTI_FEAT_FILE, FP_FEAT_FILE, use_folds=valid_folds, train=False)
                    train_loader = FastTensorDataLoader(*train_data.tensor_samples(), batch_size=args.batch,
                                                        shuffle=True)
                    valid_loader = FastTensorDataLoader(*valid_data.tensor_samples(), batch_size=len(valid_data) // 4)
                    model = create_model(train_data, hs, gpu_id)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    logging.info(
                        "Start inner loop: train folds {}, valid folds {}".format(inner_trn_folds, valid_folds))
                    ret = train_model(model, optimizer, loss_func, train_loader, valid_loader,
                                      args.epoch, args.patience, gpu_id, sl=False)
                    ret_vals.append(ret)
                    del model

                inner_loss = sum(ret_vals) / len(ret_vals)
                logging.info("Inner loop completed. Mean valid loss: {:.4f}".format(inner_loss))
                logging.info("-" * n_delimiter)
                losses.append(inner_loss)

        torch.cuda.memory_summary(device=None, abbreviated=False)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(10)
        min_ls, min_idx = arg_min(losses)
        best_hs, best_lr = param[min_idx]
        train_data = FastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, CELL_FEAT_FILE,
                                        SYNERGY_FILE, DTI_FEAT_FILE, FP_FEAT_FILE, use_folds=outer_trn_folds)
        test_data = FastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, CELL_FEAT_FILE,
                                       SYNERGY_FILE, DTI_FEAT_FILE, FP_FEAT_FILE, use_folds=[test_fold], train=False)
        model = create_model(train_data, best_hs, gpu_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

        logging.info("Best hidden size: {} | Best learning rate: {}".format(best_hs, best_lr))
        logging.info("Start test on fold {}.".format([test_fold]))
        test_mdl_dir = os.path.join(out_dir, str(test_fold))
        if not os.path.exists(test_mdl_dir):
            os.makedirs(test_mdl_dir)
        test_loss, acc, recall = eval_model(model, optimizer, loss_func, train_data, test_data,
                               args.batch, args.epoch, args.patience, gpu_id, test_mdl_dir)
        test_losses.append(test_loss)
        logging.info("Test loss: {:.4f}".format(test_loss))
        logging.info("Test accuracy: {:.4f}".format(acc))
        logging.info("Test recall: {:.4f}".format(recall))
        logging.info("*" * n_delimiter + '\n')
    logging.info("CV completed")
    with open(test_loss_file, 'wb') as f:
        pickle.dump(test_losses, f)
    mu, sigma = calc_stat(test_losses)
    logging.info("MSE: {:.4f} ± {:.4f}".format(mu, sigma))
    lo, hi = conf_inv(mu, sigma, len(test_losses))
    logging.info("Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi))
    rmse_loss = [x ** 0.5 for x in test_losses]
    mu, sigma = calc_stat(rmse_loss)
    logging.info("RMSE: {:.4f} ± {:.4f}".format(mu, sigma))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500, help="n epoch")
    parser.add_argument('--batch', type=int, default=256, help="batch size")
    parser.add_argument('--gpu', type=int, default=None, help="cuda device")
    parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
    parser.add_argument('--suffix', type=str, default=time_str, help="model dir suffix")
    parser.add_argument('--hidden', type=int, nargs='+', default=[2048, 4096, 8192], help="hidden size")
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-3, 1e-4, 1e-5], help="learning rate")
    args = parser.parse_args()
    out_dir = os.path.join(OUTPUT_DIR, 'cv_{}'.format(args.suffix))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = os.path.join(out_dir, 'cv.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)
    cv(args, out_dir)


if __name__ == '__main__':
    main()
