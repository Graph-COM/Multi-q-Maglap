import torch
import numpy as np
import wandb
from torch_geometric.loader import DataLoader
from torch import optim
from torch import nn
import argparse
import wandb
import random
from data_utils.sorting_dataset import SortingDataset, symmetrize_transform, bidirect_transform
from model.gnns import Id, Id_PE, GINE, GINEInvariant, GPSInvariant, GPS, GINEEquivariant
from model.pe_encoders import PEEncoder, NaivePEEncoder
from model.dag_transformer.models import GraphTransformer as DAGformer
from model.sat.models import GraphTransformer as SAT
from model.graph_classfication_model_sorting import GraphClassifier
from utils.get_mag_lap import AddLaplacianEigenvectorPE, AddMagLaplacianEigenvectorPE
from utils.eval_metric import generate_cross_entropy_mask
from torch_geometric.transforms import Compose
from model.dag_transformer.data import dag_pretransform
import os.path as osp
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score


class Trainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = f"cuda:{cfg.gpu_id}"

        # positional encodings pre_transform
        processed_suffix = ''
        if cfg.pe is not None and cfg.pe_dim > 0:
            processed_suffix += cfg.pe + str(cfg.pe_dim)
            if cfg.pe == 'lap':
                pre_transform = AddLaplacianEigenvectorPE(k=cfg.pe_dim, attr_name='pe')
            elif cfg.pe == 'maglap':
                processed_suffix += '_' + str(cfg.q_dim) + 'q'+str(cfg.q)
                processed_suffix += '_dynamic' if cfg.dynamic_q else ''
                pre_transform = AddMagLaplacianEigenvectorPE(k=cfg.pe_dim, q=cfg.q,
                                                         multiple_q=cfg.q_dim, attr_name='pe', dynamic_q=cfg.dynamic_q)
            else:
                raise Exception("args.pe: unknown positional encodings")
        else:
            pre_transform = None


        if cfg.base_gnn.startswith('dag'):
            processed_suffix += '_dag'
            if pre_transform is None:
                pre_transform = lambda data: dag_pretransform(data)
            else:
                pre_transform = Compose([pre_transform, lambda data: dag_pretransform(data)])

        if cfg.direct == 'un':
            transform = symmetrize_transform
        elif cfg.direct == 'bi':
            transform = bidirect_transform
        else:
            raise Exception('Unrecognized args.direct!')

        # data filtering
        n_filter = cfg.max_num_nodes
        if n_filter == -1:
            pre_filter = None
        else:
            pre_filter = lambda data: data.num_nodes <= n_filter
            processed_suffix += '_n' + str(n_filter)
        # pre_filter = None

        # load dataset with filtering
        train_dataset = SortingDataset(name='sorting/7to11_12_13to16', root='data/', transform=transform,
                                        pre_filter=pre_filter, pre_transform=pre_transform,
                                        processed_suffix=processed_suffix, split='train')
        val_dataset = SortingDataset(name='sorting/7to11_12_13to16', root='data/', transform=transform,
                                        pre_filter=pre_filter, pre_transform=pre_transform,
                                        processed_suffix=processed_suffix, split='valid')
        test_dataset = SortingDataset(name='sorting/7to11_12_13to16', root='data/', transform=transform,
                                      pre_filter=pre_filter, pre_transform=pre_transform,
                                      processed_suffix=processed_suffix, split='test')

        self.train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        # construct model
        actual_pe_dim = cfg.pe_dim
        if cfg.pe == 'maglap':
            actual_pe_dim *= 2 * cfg.q_dim
        #if cfg.base_gnn == 'transformer':
        #    base_gnn = GraphTransformer(cfg.node_emb_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.num_layers, norm=cfg.bn)
        #elif cfg.base_gnn == 'transformer_e':
        #    base_gnn = GraphTransformerInvariant(cfg.node_emb_dim, cfg.hidden_dim, cfg.hidden_dim,
                                                 #actual_pe_dim, cfg.q_dim, cfg.num_layers, norm=cfg.bn,
                                                 #handle_symmetry=cfg.handle_symmetry, pe_type=cfg.pe)
        # pe encoder: a learnable pre-processing model for pe
        pe_config = {'pe_dim': cfg.pe_dim, 'q_dim': cfg.q_dim, 'pe_type': cfg.pe, 'pe_norm': cfg.pe_norm}
        if cfg.pe_encoder == 'naive':
            pe_encoder = NaivePEEncoder(pe_config, cfg.hidden_dim)
        elif cfg.pe_encoder == 'none':
            pe_encoder = None
        else:
            pe_encoder = PEEncoder(pe_config, cfg.hidden_dim, encoder=cfg.pe_encoder, sign_inv=cfg.sign_inv,
                                   attn=cfg.pe_attn, dropout=0.15)

        # graph encoder
        if cfg.base_gnn == 'gin':
            base_gnn = GINE(cfg.hidden_dim, cfg.num_layers)
        elif cfg.base_gnn == 'gin_e':
            base_gnn = GINEInvariant(cfg.hidden_dim, actual_pe_dim, cfg.q_dim, cfg.num_layers, pe_type=cfg.pe,
                                     handle_symmetry=cfg.handle_symmetry)
        elif cfg.base_gnn.startswith('transformer'):
            if cfg.base_gnn.endswith('_e'):
                pe_config['handle_symmetry'] = cfg.handle_symmetry
            else:
                pe_config = None
            # use a 0-hop SAT to do transformer
            base_gnn = SAT(d_model=cfg.hidden_dim,
                             dim_feedforward=4*cfg.hidden_dim,
                             dropout=0.2,
                             num_heads=4,
                             num_layers=cfg.num_layers,
                             batch_norm=True,
                             gnn_type='gcn',
                             k_hop=0,
                             se='gnn',
                             deg=None,
                             edge_dim=cfg.hidden_dim,
                             pe_config=pe_config,
                           )
        elif cfg.base_gnn == 'none':
            base_gnn = Id(cfg.hidden_dim)
        elif cfg.base_gnn == 'none_e':
            base_gnn = Id_PE(cfg.hidden_dim)

        # overall network
        #if '_e' in cfg.base_gnn or '_eq' in cfg.base_gnn:
        #    self.predictor = GraphClassifierInvariant(cfg.node_emb_dim, actual_pe_dim, cfg.q_dim,
                                                     #gnn_model=base_gnn, pe_type=cfg.pe)
        #else:
        self.predictor = GraphClassifier(cfg.node_emb_dim, gnn_model=base_gnn, pe_model=pe_encoder)
        self.predictor.to(self.device)

        print(f'#Params: {sum(p.numel() for p in self.predictor.parameters())}')

        # construct optimizer
        if cfg.optimizer == 'adam':
            self.optimizer = optim.Adam(self.predictor.parameters(), betas=(0.7, 0.9), weight_decay=6e-5, lr=cfg.lr)
        elif cfg.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.predictor.parameters(), betas=(0.7, 0.9), lr=cfg.lr, weight_decay=6e-5)

        # construct scheduler
        if cfg.scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.epochs - cfg.warmup)
        else:
            self.scheduler = None

        # warm-up strategy
        if cfg.warmup > 0:
            lr_steps = cfg.lr / (cfg.warmup * len(self.train_loader))
            def warmup_lr_scheduler(s):
                lr = s * lr_steps
                return lr
            self.warmup_scheduler = warmup_lr_scheduler

        # training and evaluation loss
        #self.loss = nn.L1Loss(reduction='mean')
        #self.loss = nn.MSELoss(reduction='mean')
        #self.loss = nn.CrossEntropyLoss(reduction='none')
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')


        #test_maglap(train_dataset, self.predictor, dist='lpd')

        # init wandb
        if cfg.wandb:
            raise Exception('init wandb or disable it')


    def train(self):
        best_val_loss, best_test_loss = 9999.0, 9999.0
        #self.predictor.load_state_dict(torch.load('./model_code.pt'))

        for curr_epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(curr_epoch)
            val_metrics, val_ce_loss = self.evaluate(self.val_loader)
            test_metrics, test_ce_loss = self.evaluate(self.test_loader)
            val_loss = 1. - val_metrics['overall']['F1']
            test_loss = 1. - test_metrics['overall']['F1']
            if curr_epoch > self.cfg.warmup and self.scheduler is not None:
                self.scheduler.step()
            #val_loss, test_loss = 0., 0.
            # self.scheduler.step(eval_loss)
            # lr = self.scheduler.get_last_lr()[0]
            #lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if self.cfg.wandb:
                report = {'train_loss': train_loss, 'val_f1': val_metrics['overall']['F1'],
                          'test_f1': test_metrics['overall']['F1'], 'val_loss': val_ce_loss, 'test_loss': test_ce_loss}
                wandb.log(report)
            if val_loss < best_val_loss:
                best_val_loss, best_test_loss = val_loss, test_loss
                if self.cfg.wandb:
                    wandb.run.summary["best_val_f1"] = val_metrics['overall']['F1']
                    for key in test_metrics:
                        for metric in test_metrics[key]:
                            wandb.run.summary['best_test_'+key+'_'+metric] = test_metrics[key][metric]
                    wandb.run.summary["best_training_loss"] = train_loss
                    wandb.run.summary['best_epoch'] = curr_epoch
        print('Best test loss: %.6f' % best_test_loss)

    def train_epoch(self, curr_epoch):
        self.predictor.train()
        total_loss = 0
        total_acc = 0
        print('Training Epoch %d...' % curr_epoch)
        for i, batch in enumerate(self.train_loader):
            if i % 5 == 0:
                print('Training Batch %d / %d' % (i, len(self.train_loader)))
            if curr_epoch <= self.cfg.warmup:
                iteration = (curr_epoch - 1) * len(self.train_loader) + i + 1
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.warmup_scheduler(iteration)
            loss = self.train_batch(batch)
            total_loss += loss
            # total_acc += acc
        ave_loss = total_loss / len(self.train_loader.dataset)
        # ave_acc = total_acc / len(self.train_loader.dataset)
        #print('Training Epoch %d: Loss %.3f' % (curr_epoch, ave_loss))
        return ave_loss

    def train_batch(self, batch):
        batch.to(self.device)
        self.optimizer.zero_grad()
        loss = 0.
        acc = 0.
        y_pred = self.predictor(batch)[:, 0]  # [B]
        y = batch.y
        loss = self.loss(y_pred, y)


            # acc += (y_pred_i.argmax(-1) == y[:, i]).float().sum()
        # loss = loss / len(y_pred)
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        # acc = acc.item() / 5
        # self.scheduler.step()

        return loss * y.size(0)

    def evaluate(self, eval_loader):
        self.predictor.eval()
        pred_list = []
        ref_list = eval_loader.dataset.y.tolist()
        seq_len_list = eval_loader.dataset.seq_len[:, 0].tolist()
        total_loss = 0
        for batch in eval_loader:
            pred = self.evaluate_batch(batch)
            total_loss += self.loss(pred, batch.y)
            pred_list += (pred > 0.).float().tolist()
            #ref_list += batch.y.tolist()
            #seq_len_list += batch.seq_len[:, 0].tolist()
        return self.eval_metrics(pred_list, ref_list, seq_len_list), total_loss / len(eval_loader)

    def evaluate_batch(self, batch):
        batch.to(self.device)
        #loss = 0.
        with torch.no_grad():
            pred = self.predictor(batch)[:, 0]
            #pred = (pred > 0.).float()

        #return pred.tolist()
        return pred


    def eval_metrics(self, pred_list, ref_list, seq_len_list):
        pred, ref, seq_len = np.array(pred_list), np.array(ref_list), np.array(seq_len_list)
        seq_len_unique = np.unique(seq_len)
        report = {'overall': {'precision': precision_score(ref, pred), 'recall': recall_score(ref, pred), 'F1': f1_score(ref, pred)}}
        for seq_l in seq_len_unique:
            ind = np.where(seq_len == seq_l)
            pred_l = pred[ind]
            ref_l = ref[ind]
            report['seq_len_%d' % seq_l] = {'precision': precision_score(ref_l, pred_l), 'recall': recall_score(ref_l, pred_l),
                    'F1': f1_score(ref_l, pred_l)}
        return report


class Config:
    def __init__(self, args):
        for key, value in args._get_kwargs():
            setattr(self, key, value)

def set_seed(seed: int) -> None:
    """
    Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_utils.py#L83
    """
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    torch.set_num_threads(2)
    # hyper-parameters parsing
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--num_node_types", type=int, default=0)
    parser.add_argument("--max_num_nodes", type=int, default=-1)
    parser.add_argument("--subset", type=int, default=-1)
    # training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--wandb", action='store_true',default=False)
    parser.add_argument("--direct", type=str, default='bi') # un, uni, bi, bi-dfs
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--scheduler", action='store_true', default=True)
    # model parameters
    parser.add_argument("--base_gnn", type=str, default='gin')
    parser.add_argument("--node_emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--pe", type=str, default=None)
    parser.add_argument("--pe_dim", type=int, default=0)
    parser.add_argument("--degree", action='store_true', default=False)
    # parameters for maglap
    parser.add_argument("--q", type=float, default=1e-2)
    parser.add_argument("--q_dim", type=int, default=1)
    parser.add_argument("--dynamic_q", action='store_true', default=False)
    parser.add_argument("--handle_symmetry", type=str, default='spe')
    parser.add_argument("--pe_encoder", type=str, default='naive')
    parser.add_argument("--pe_attn", action='store_true', default=False)
    parser.add_argument("--pe_norm", action='store_true', default=False)
    parser.add_argument("--sign_inv", action='store_true', default=False)
    args = parser.parse_args()
    cfg = Config(args)
    trainer = Trainer(cfg)
    trainer.train()



if __name__ == "__main__":
    main()
