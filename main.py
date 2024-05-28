import torch
import wandb
from torch_geometric.loader import DataLoader
from torch import optim
from torch import nn
import argparse
import wandb
import random
from data_utils.distance_dataset import random_graph_distance
from model.gnns import Id, GIN, GraphTransformer, GraphTransformerInvariant, Id_PE
from model.link_prediction_model import LinkPrediction, LinkPredictionInvariant
from model.pe_encoders import NaivePEEncoder, PEEncoder
from utils.get_mag_lap import AddLaplacianEigenvectorPE, AddMagLaplacianEigenvectorPE, AddSingularValuePE
from torch_geometric.utils import degree


from test_utils import test_maglap


def my_transform(data):
    d = torch.cat([degree(data.edge_index[0], data.num_nodes).unsqueeze(-1),
               degree(data.edge_index[1], data.num_nodes).unsqueeze(-1)], dim=-1)

    data.update({"degree": d})
    return data

class Trainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = f"cuda:{cfg.gpu_id}"

        # construct dataset
        transform = my_transform
        processed_suffix = ''
        if cfg.pe is not None and cfg.pe_dim > 0:
            processed_suffix += cfg.pe + str(cfg.pe_dim)
            if cfg.pe == 'lap':
                pre_transform = AddLaplacianEigenvectorPE(k=cfg.pe_dim, attr_name='pe')
            elif cfg.pe == 'svd':
                pre_transform = AddSingularValuePE(k=cfg.pe_dim, attr_name='pe')
            elif cfg.pe == 'maglap':
                processed_suffix += '_' + str(cfg.q_dim) + 'q'+str(cfg.q)
                processed_suffix += '_dynamic' if cfg.dynamic_q else ''
                pre_transform = AddMagLaplacianEigenvectorPE(k=cfg.pe_dim, q=cfg.q,
                                                         dynamic_q=cfg.dynamic_q,
                                                         multiple_q=cfg.q_dim, attr_name='pe')
            else:
                raise Exception("args.pe: unknown positional encodings")
        else:
            pre_transform = None
        train_dataset = random_graph_distance(dataname=cfg.dataname, root='./data', split="train",
                                           pre_transform=pre_transform,
                                           transform=transform, processed_suffix=processed_suffix)
        test_dataset = random_graph_distance(dataname=cfg.dataname, root='./data', split="valid",
                                          pre_transform=pre_transform,
                                          transform=transform, processed_suffix=processed_suffix)
        
        # construct dataloader
        # use subset of training set
        train_id = [i for i in range(len(train_dataset))]
        random.shuffle(train_id)
        if cfg.subset != -1:
            train_id = train_id[:cfg.subset]
        train_id, val_id = train_id[:int(len(train_id)*0.95)], train_id[int(len(train_id)*0.95):]
        self.train_loader = DataLoader(train_dataset[train_id], batch_size=cfg.batch_size, shuffle=True)
        self.val_loader = DataLoader(train_dataset[val_id], batch_size=cfg.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

        # construct model
        # pe encoder: a learnable pre-processing model for pe
        pe_config = {'pe_dim': cfg.pe_dim, 'q_dim': cfg.q_dim, 'pe_type': cfg.pe, 'pe_norm': False}
        if cfg.pe_encoder:
            pe_encoder = PEEncoder(pe_config, cfg.hidden_dim, encoder='mlp', sign_inv=True)
        else:
            pe_encoder = None


        # base_gnn = GIN(cfg.node_emb_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.num_layers)
        actual_pe_dim = cfg.pe_dim
        eigval_dim = cfg.pe_dim * cfg.q_dim
        if cfg.pe == 'maglap':
            actual_pe_dim *= 2 * cfg.q_dim
        elif cfg.pe == 'svd':
            actual_pe_dim *= 2
            eigval_dim = cfg.pe_dim
        if cfg.base_gnn == 'transformer':
            base_gnn = GraphTransformer(cfg.hidden_dim, cfg.num_layers)
        elif cfg.base_gnn == 'transformer_e':
            base_gnn = GraphTransformerInvariant(cfg.hidden_dim,
                                                 actual_pe_dim, cfg.q_dim, cfg.num_layers,
                                                 handle_symmetry=cfg.handle_symmetry, pe_type=cfg.pe)
        elif cfg.base_gnn == 'none':
            base_gnn = Id(cfg.hidden_dim)
        elif cfg.base_gnn == 'none_e':
            base_gnn = Id_PE(cfg.hidden_dim)
        if '_e' in cfg.base_gnn:
            self.predictor = LinkPredictionInvariant(cfg.num_node_types, cfg.node_emb_dim, actual_pe_dim,
                                                     cfg.q_dim, base_gnn, pe_type=cfg.pe,
                                                     handle_symmetry=cfg.handle_symmetry, out_dim=cfg.out_dim)
        else:
            self.predictor = LinkPrediction(cfg.num_node_types, cfg.node_emb_dim, actual_pe_dim, eigval_dim, base_gnn,
                                            pe_model=pe_encoder, out_dim=cfg.out_dim)
        self.predictor.to(self.device)

        # construct optimizer
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=cfg.lr)

        # training and evaluation loss
        #self.loss = nn.L1Loss(reduction='mean')
        self.loss = nn.MSELoss(reduction='mean')

        #test_maglap(train_dataset, self.predictor, dist='lpd')
        #rint('clear')

        # init wandb
        if cfg.wandb:
            raise Exception('init wandb or disable it')


    def train(self):
        best_val_loss, best_test_loss = 9999.0, 9999.0

        for curr_epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(curr_epoch)
            val_loss = self.evaluate(self.val_loader)
            test_loss = self.evaluate(self.test_loader)
            # self.scheduler.step(eval_loss)
            # lr = self.scheduler.get_last_lr()[0]
            #lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if self.cfg.wandb:
                wandb.log({'train_loss': train_loss, 'eval_loss': val_loss, 'test_loss': test_loss})
            if val_loss < best_val_loss:
                best_val_loss, best_test_loss = val_loss, test_loss
                if self.cfg.wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_test_loss"] = best_test_loss
                    wandb.run.summary["best_training_loss"] = train_loss
                    wandb.run.summary['best_epoch'] = curr_epoch
        print('Best test loss: %.6f' % best_test_loss)
        #torch.save(self.predictor.state_dict(), 'model.pt')

    def train_epoch(self, curr_epoch):
        self.predictor.train()
        total_loss = 0
        print('Training Epoch %d...' % curr_epoch)
        for i, batch in enumerate(self.train_loader):
            total_loss += self.train_batch(batch)
        ave_loss = total_loss / self.train_loader.dataset.y.size(0)
        #print('Training Epoch %d: Loss %.3f' % (curr_epoch, ave_loss))
        return ave_loss

    def train_batch(self, batch):
        batch.to(self.device)
        self.optimizer.zero_grad()

        y_pred = self.predictor(batch)  # [B]
        #loss = self.loss(y_pred.view(-1), batch.y)  # [1]
        loss = self.loss(y_pred, batch.y.view(y_pred.size()))  # [1]
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        # self.scheduler.step()

        return loss * batch.y.size(0)

    def evaluate(self, eval_loader):
        self.predictor.eval()
        total_loss = 0.0
        for batch in eval_loader:
            total_loss += self.evaluate_batch(batch)
        total_loss /= eval_loader.dataset.y.size(0)
        return total_loss

    def evaluate_batch(self, batch):
        batch.to(self.device)
        with torch.no_grad():
            y_pred = self.predictor(batch)
        #return self.loss(y_pred.view(-1), batch.y).item() * batch.y.size(0)
        return self.loss(y_pred, batch.y.view(y_pred.size())).item() * batch.y.size(0)


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
    parser.add_argument("--dataname", type=str, default='distance/16to63_64to71_72to83_ca')
    parser.add_argument("--num_node_types", type=int, default=0)
    parser.add_argument("--subset", type=int, default=-1)
    # training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--wandb", action='store_true',default=False)
    # model parameters
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--base_gnn", type=str, default='none')
    parser.add_argument("--node_emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--pe", type=str, default=None)
    parser.add_argument("--pe_dim", type=int, default=0)
    parser.add_argument("--bn", action='store_true', default=False)
    parser.add_argument("--pe_encoder", action='store_true', default=False)
    # parameters for maglap
    parser.add_argument("--q", type=float, default=1e-2)
    parser.add_argument("--dynamic_q", action='store_true', default=False)
    parser.add_argument("--q_dim", type=int, default=1)
    parser.add_argument("--handle_symmetry", type=str, default='spe')
    args = parser.parse_args()
    cfg = Config(args)
    trainer = Trainer(cfg)
    trainer.train()



if __name__ == "__main__":
    main()
