import numpy as np
import wandb
import shutil
import yaml
import importlib
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import r2_score
import pickle

import torch

from torchmetrics.regression import MeanAbsolutePercentageError

from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch_geometric.utils import to_undirected

import ray
from ray import train, tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
from hyperopt import hp

from utils import exception_not_defined, create_nested_folder, find_latest_model, delete_file_with_head

class AMPRunner():
    def __init__(self, config):
        self.config = config
        self.task = config['task']
        self.train_folder = config['train']['train_files']+str(config['task']['name'])+ '_' +str(config['task']['type'])+'/'+str(config['task']['target'])+'_'+str(config['model'].get('pe_file_name'))+'/'+str(config['model']['name'])+'/'
        self.result_csv = config['train']['train_files']+str(config['task']['name'])+ '_' +str(config['task']['type'])+'/'+str(config['task']['target'])+'_'+str(config['model'].get('pe_file_name'))+'/'+str(config['model']['name'])+'/result.csv'
        create_nested_folder(self.train_folder)
        # define the loss criterion
        if self.config['train']['criterion'] == 'L1':
            self.criterion = nn.L1Loss()
        elif self.config['train']['criterion'] == 'SmoothL1':  
            self.criterion = nn.SmoothL1Loss()
        elif self.config['train']['criterion'] == 'MSE':
            self.criterion = nn.MSELoss()
        else: 
            exception_not_defined('criterion')
        self.mse = nn.MSELoss()
        self.mape = MeanAbsolutePercentageError()
    def train_ray(self, tune_parameter_config):
        self.init_wandb(tune_parameter_config)
        # initialize the datasets and dataloader
        module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
        attribute_name = self.config['task']['name']+'DataProcessor'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.train_data = cls(self.config, 'train')
        self.val_data = cls(self.config, 'valid')
        batch_size = tune_parameter_config['batch_size']
        self.train_loader = DataLoader(self.train_data, batch_size = batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_data, batch_size = batch_size, shuffle = False)

        # define the model, optimizer, and scheduler here
        module_path = 'models.'+self.config['task']['name'] + '.' + self.config['task']['name']+'_model'
        attribute_name = self.config['task']['name'] + 'Model'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.model = cls(tune_parameter_config)
        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=float(tune_parameter_config['lr']), betas=(0.9, 0.999))
        else:
            exception_not_defined('optimizer')
        if self.config['train']['scheduler']['name'] == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=self.config['train']['scheduler']['step_size'], gamma=self.config['train']['scheduler']['gamma'])
        else: 
            exception_not_defined('scheduler')
        #self.device = self.config['train']['device']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.model.train()
        self.best_valid_metric = 10000
        self.mape.to(self.device)

        # TO DO: use best val loss as reported metric for tuning
        for epoch_idx in tqdm(range(self.config['train']['epoch'])):
            train_loss, train_rmse, train_r2 = self.train_one_epoch(self.train_loader, 'train', epoch_idx)
            valid_loss, valid_rmse, valid_r2 = self.train_one_epoch(self.val_loader, 'valid', epoch_idx)
            self.scheduler.step()
            #self.save_model(valid_loss, epoch_idx)
            if valid_loss < self.best_valid_metric:
                self.best_valid_metric = valid_loss
            train.report({'hidden_dim': tune_parameter_config['hidden_dim'], 'num_layer': tune_parameter_config['num_layers'],
                          'mse' : self.best_valid_metric, 'r2' : valid_r2, 'batch_size': tune_parameter_config['batch_size'],
                          'lr': tune_parameter_config['lr'],'dropout': tune_parameter_config['dropout'], 
                          'mlp_out': tune_parameter_config['mlp_out']['num_layer'],
                        })
    
    def train(self):
        self.init_wandb()
        # initialize the datasets and dataloader
        module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
        attribute_name = self.config['task']['name']+'DataProcessor'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.train_data = cls(self.config, 'train')
        self.val_data = cls(self.config, 'valid')
        batch_size = self.config['train']['batch_size']
        self.train_loader = DataLoader(self.train_data, batch_size = batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_data, batch_size = batch_size, shuffle = False)
        stat_path = str(self.config['task']['processed_folder'])+str(self.config['task']['name'])+'/stats.pkl'
        with open(stat_path, 'rb') as file:
            self.statistics = pickle.load(file)
        # define the model, optimizer, and scheduler here
        module_path = 'models.'+str(self.config['task']['name']) + '.' + self.config['task']['name']+'_model'
        attribute_name = self.config['task']['name']+'Model'
        module = importlib.import_module(module_path)
        
        cls = getattr(module, attribute_name)
        self.model = cls(self.config['model'])
        #self.model = cls(self.config['model']['args'])
        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.config['train']['lr']), betas=(0.9, 0.999))
        else:
            exception_not_defined('optimizer')
        if self.config['train']['scheduler']['name'] == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=self.config['train']['scheduler']['step_size'], gamma=self.config['train']['scheduler']['gamma'])
        else: 
            exception_not_defined('scheduler')
        self.device = int(self.config['train']['device'])
        self.model.to('cuda:'+str(self.device) if torch.cuda.is_available() else 'cpu')
        self.model.train()
        self.best_valid_metric = 10000

        # initialize the normalization of regression
        
        for epoch_idx in tqdm(range(self.config['train']['epoch'])):
            train_loss, train_rmse, train_r2 = self.train_one_epoch(self.train_loader, 'train', epoch_idx)
            valid_loss, valid_rmse, valid_r2 = self.train_one_epoch(self.val_loader, 'valid', epoch_idx)
            self.scheduler.step()
            self.save_model(valid_loss, epoch_idx)

    def train_one_epoch(self, data_loader, mode, epoch_idx):
        epoch_loss = 0
        epoch_gain_loss = 0
        epoch_pm_loss = 0
        epoch_bw_loss = 0
        total_graphs = 0
        pred_gain_list = []
        pred_pm_list = []
        pred_bw_list = []
        y_gain_list = []
        y_pm_list = []
        y_bw_list = []
        for batch_data in data_loader:
            batch_data.to(self.device)
            if self.config['train']['directed'] == 0:
                all_edge_index = to_undirected(batch_data.edge_index)
                sub_edge_index = to_undirected(batch_data.sub_edge_index)
                batch_data.edge_index = all_edge_index
                batch_data.sub_edge_index = sub_edge_index
            y_gain = getattr(batch_data, 'gain', None)
            y_pm = getattr(batch_data, 'pm', None)
            y_bw = getattr(batch_data, 'bw', None)
            '''y_gain = (y_gain - self.statistics['gain_mean']) / self.statistics['gain_std']
            y_pm = (y_pm - self.statistics['pm_mean']) / self.statistics['pm_std']
            y_bw = (y_pm - self.statistics['bw_mean']) / self.statistics['bw_std']'''
            if mode == 'train':
                self.model.train()
                self.optimizer.zero_grad()
                if self.config['model']['name'] in ['GPS', 'GPSSE']:
                    self.model.middle_model1.redraw_projection.redraw_projections()
                    self.model.middle_model2.redraw_projection.redraw_projections()
                    pred_gain, pred_pm, pred_bw = self.model(batch_data)
                else:
                    pred_gain, pred_pm, pred_bw = self.model(batch_data)
            elif mode == 'valid':
                self.model.eval()
                with torch.no_grad():
                    pred_gain, pred_pm, pred_bw = self.model(batch_data)
            pred_gain_list.append(pred_gain.detach().cpu().numpy().reshape(-1))
            pred_pm_list.append(pred_pm.detach().cpu().numpy().reshape(-1))
            pred_bw_list.append(pred_bw.detach().cpu().numpy().reshape(-1))
            y_gain_list.append(y_gain.detach().cpu().numpy().reshape(-1))
            y_pm_list.append(y_pm.detach().cpu().numpy().reshape(-1))
            y_bw_list.append(y_bw.detach().cpu().numpy().reshape(-1))
            gain_loss = self.criterion(pred_gain, y_gain.reshape(-1, 1))
            pm_loss = self.criterion(pred_pm, y_pm.reshape(-1, 1))
            bw_loss = self.criterion(pred_bw, y_bw.reshape(-1, 1))
            
            #batch_loss = gain_loss + pm_loss + bw_loss
            if self.config['task']['target'] == 'gain':
                batch_loss = gain_loss
            elif self.config['task']['target'] == 'pm':
                batch_loss = pm_loss
            elif self.config['task']['target'] == 'bw':
                batch_loss = bw_loss

            if mode == 'train':
                batch_loss.backward()
                self.optimizer.step()
            epoch_loss = epoch_loss + batch_loss.item() * (torch.max(batch_data.batch).item() + 1)
            epoch_gain_loss = epoch_gain_loss + gain_loss.item() * (torch.max(batch_data.batch).item() + 1)
            epoch_pm_loss = epoch_pm_loss + pm_loss.item() * (torch.max(batch_data.batch).item() + 1)
            epoch_bw_loss = epoch_bw_loss + bw_loss.item() * (torch.max(batch_data.batch).item() + 1)
            total_graphs = total_graphs + torch.max(batch_data.batch).item() + 1
        epoch_loss = epoch_loss / total_graphs
        epoch_gain_rmse = np.sqrt(epoch_gain_loss / total_graphs)
        epoch_pm_rmse = np.sqrt(epoch_pm_loss / total_graphs)
        epoch_bw_rmse = np.sqrt(epoch_bw_loss / total_graphs)
        epoch_rmse = epoch_gain_rmse + epoch_pm_rmse + epoch_bw_rmse
        
        pred_gain_list = np.concatenate(pred_gain_list)
        pred_pm_list = np.concatenate(pred_pm_list)
        pred_bw_list = np.concatenate(pred_bw_list)
        y_gain_list = np.concatenate(y_gain_list)
        y_pm_list = np.concatenate(y_pm_list)
        y_bw_list = np.concatenate(y_bw_list)
        epoch_gain_r2 = r2_score(pred_gain_list, y_gain_list)
        epoch_pm_r2 = r2_score(pred_pm_list, y_pm_list)
        epoch_bw_r2 = r2_score(pred_bw_list, y_bw_list)
        epoch_r2 = (epoch_gain_r2 + epoch_pm_r2 + epoch_bw_r2) / 3
        self.write_log({'loss': epoch_loss, 'total rmse': epoch_rmse, 'gain rmse': epoch_gain_rmse, 
                        'pm rmse': epoch_pm_rmse, 'bw rmse': epoch_bw_rmse,
                        'gain r2': epoch_gain_r2, 'pm r2': epoch_pm_r2, 'bw r2': epoch_bw_r2,
                        'epoch r2': epoch_r2,
                        }, epoch_idx, mode)
        return epoch_loss, epoch_rmse, epoch_r2
     
    def test(self, load_statedict = True, test_num_idx = 0):
        if load_statedict:
            self.device = int(self.config['train']['device'])
            module_path = 'models.'+str(self.config['task']['name']) + '.' + self.config['task']['name']+'_model'
            attribute_name = self.config['task']['name']+'Model'
            module = importlib.import_module(module_path)
            cls = getattr(module, attribute_name)
            self.model = cls(self.config['model'])
            dict_path = find_latest_model(self.train_folder, 'model')
            state_dict = torch.load(dict_path) 
            self.model.load_state_dict(state_dict)
            self.model = self.model.to('cuda:'+str(self.device) if torch.cuda.is_available() else 'cpu')
            self.model.eval()
        else:
            #pass
            self.model.eval() 
        # start testing
        test_list = ['id', 'ood']
        self.test_data_dict = {}
        self.test_loader_dict = {}
        for test_name in test_list:
            module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
            attribute_name = self.config['task']['name']+'DataProcessor'
            module = importlib.import_module(module_path)
            cls = getattr(module, attribute_name)
            test_data = cls(self.config, 'test_' + test_name)
            self.test_data_dict[test_name] = test_data
            test_loader = DataLoader(test_data, batch_size = 1, shuffle = False)
            self.test_loader_dict[test_name] = test_loader
        table = PrettyTable(['test set name', '# of samples', 'gain mse', 'gain rmse', 'gain r2', 'pm mse', 
                             'pm rmse', 'pm r2', 'bw mse', 'bw rmse', 'bw r2'])
        for test_name in test_list:
            gain_mse, gain_rmse, gain_r2, pm_mse, pm_rmse, pm_r2, bw_mse, bw_rmse, bw_r2 = self.test_a_task(test_name)
            if test_name == 'id':
                row = ['stage 3', str(len(self.test_data_dict[test_name])), gain_mse, gain_rmse, gain_r2, pm_mse, pm_rmse, pm_r2, bw_mse, bw_rmse, bw_r2]
                table.add_row(row)
            elif test_name == 'ood':
                row = ['stage 2', str(len(self.test_data_dict[test_name])), gain_mse, gain_rmse, gain_r2, pm_mse, pm_rmse, pm_r2, bw_mse, bw_rmse, bw_r2]
                table.add_row(row)
            if self.config['train']['wandb'] == 1:
                wandb.run.summary[test_name + '-gain_rmse'] = gain_rmse
                wandb.run.summary[test_name + '-gain_r2'] = gain_r2
                wandb.run.summary[test_name + '-pm_rmse'] = pm_rmse
                wandb.run.summary[test_name + '-pm_r2'] = pm_r2
                wandb.run.summary[test_name + '-bw_rmse'] = bw_rmse
                wandb.run.summary[test_name + '-bw_r2'] = bw_r2

        if test_num_idx == 0:
            with open(self.result_csv, 'w', newline='') as f_output:
                f_output.write(table.get_csv_string())
        else:
            with open(self.result_csv, 'a', newline='') as f_output:
                f_output.write(table.get_csv_string())
                
        print(table)
        
    def test_a_task(self, testset_name):
        pred_gain_list = []
        y_gain_list = []
        pred_pm_list = []
        y_pm_list = []
        pred_bw_list = []
        y_bw_list = []
        for data_idx, batch_data in tqdm(enumerate(self.test_loader_dict[testset_name])):
            batch_data.to(self.device)
            if self.config['train']['directed'] == 0:
                all_edge_index = to_undirected(batch_data.edge_index)
                sub_edge_index = to_undirected(batch_data.sub_edge_index)
                batch_data.edge_index = all_edge_index
                batch_data.sub_edge_index = sub_edge_index
            y_gain = getattr(batch_data, 'gain', None)
            y_pm = getattr(batch_data, 'pm', None)
            y_bw = getattr(batch_data, 'bw', None)
            with torch.no_grad():
                pred_gain, pred_pm, pred_bw = self.model(batch_data)
            pred_gain_list.append(pred_gain)
            pred_pm_list.append(pred_pm)
            pred_bw_list.append(pred_bw)
            y_gain_list.append(y_gain)
            y_pm_list.append(y_pm)
            y_bw_list.append(y_bw)
        pred_gain_list =torch.cat(pred_gain_list, dim = 1).reshape(-1, 1)
        pred_pm_list =torch.cat(pred_pm_list, dim = 1).reshape(-1, 1)
        pred_bw_list =torch.cat(pred_bw_list, dim = 1).reshape(-1, 1)
        '''pred_gain_list = pred_gain_list * self.statistics['gain_std'] + self.statistics['gain_mean'] 
        pred_pm_list = pred_pm_list * self.statistics['pm_std'] + self.statistics['pm_mean']
        pred_bw_list = pred_pm_list * self.statistics['bw_std'] + self.statistics['bw_mean'] '''
        y_gain_list =torch.cat(y_gain_list, dim = 1).reshape(-1, 1)
        y_pm_list =torch.cat(y_pm_list, dim = 1).reshape(-1, 1)
        y_bw_list =torch.cat(y_bw_list, dim = 1).reshape(-1, 1)
        gain_loss = self.criterion(pred_gain_list, y_gain_list)
        pm_loss = self.criterion(pred_pm_list, y_pm_list)
        bw_loss = self.criterion(pred_bw_list, y_bw_list)
        gain_rmse = torch.sqrt(gain_loss)
        pm_rmse = torch.sqrt(pm_loss)
        bw_rmse = torch.sqrt(bw_loss)
        gain_r2 = r2_score(pred_gain_list.detach().cpu().numpy(), y_gain_list.detach().cpu().numpy())
        pm_r2 = r2_score(pred_pm_list.detach().cpu().numpy(), y_pm_list.detach().cpu().numpy())
        bw_r2 = r2_score(pred_bw_list.detach().cpu().numpy(), y_bw_list.detach().cpu().numpy())
        return gain_loss.item(), gain_rmse.item(), gain_r2, pm_loss.item(), pm_rmse.item(), pm_r2, bw_loss.item(), bw_rmse.item(), bw_r2

    def save_config(self):
        with open(self.train_folder + 'config.yaml', 'w') as file:
            yaml.dump(self.config, file)
        return
    def write_log(self, items, epoch_idx, mode):
        print('epoch: '+str(epoch_idx)+' '+str(mode))
        for key in items.keys():
            print(str(key) + ' ' + str(items[key]))
        if self.config['train']['wandb'] == 1:
            for key in items.keys():
                if 'loss' in key or self.config['task']['target'] in key:
                    wandb.log({mode + ' ' + str(key): items[key]}, step=epoch_idx)

    def init_wandb(self):
        if self.config['train']['wandb'] == 1:
            wandb.init(project='EDA_benchmark', name = self.config['task']['name']+'_'+str(self.config['task']['type'])+'_'+self.config['task']['target']+'_'+self.config['model']['name'])
    def save_model(self, valid_metric, epoch_idx):
        if valid_metric < self.best_valid_metric:
            self.best_valid_metric = valid_metric
            delete_file_with_head(self.train_folder, 'model')
            torch.save(self.model.state_dict(), self.train_folder+'model'+'_epoch'+str(epoch_idx)+'.pth')

    def raytune(self, tune_config, num_samples, num_cpu, num_gpu_per_trial):
        reporter = CLIReporter(parameter_columns=['hidden_dim'],metric_columns=['loss', 'mse', 'r2'])
        # init ray tune
        if self.config['model'].get('pe_file_name') in ['lap_naive', 'maglap_1q_naive'] and self.config['model']['name'] in ['GPS', 'GPSSE']:
            hidden_dim = 6 + 48 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1]))
        else: 
            hidden_dim = 48 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1]))
        dropout_p = hp.choice('dropout_p', tune_config['dropout'])
        tune_parameter_config = {
        'name': tune_config['name'],
        'batch_size': hp.choice('batch_size', tune_config['batch_size']),
        'hidden_dim': hidden_dim,
        'num_layers': hp.randint('num_layers', int(tune_config['num_layers'][0]), int(tune_config['num_layers'][1])),
        'lr': hp.uniform('lr', float(tune_config['lr'][0]), float(tune_config['lr'][1])),
        'dropout': dropout_p,
        'mlp_out': {'num_layer': hp.randint('mlp_out', int(tune_config['mlp_out']['num_layer'][0]), 
                                            int(tune_config['mlp_out']['num_layer'][1]))},
        'node_input_dim': self.config['model']['node_input_dim'],
        'edge_input_dim': self.config['model']['edge_input_dim'],
        'pe_dim_input': tune_config['pe_dim_input'],
        'pe_dim_output': tune_config['pe_dim_output'],
        'criterion': 'MSE',
        'attn_type': 'multihead',
        #'attn_kwargs': {'dropout': dropout_p}, # do not tune atten as we do not use them
        'attn_kwargs': {'dropout': 0}
        }
        tune_parameter_config = {**self.config['model'], **tune_parameter_config}
        scheduler = ASHAScheduler(
            max_t=300,
            grace_period=100,
            reduction_factor=2)
        
        hyperopt_search = HyperOptSearch(tune_parameter_config, metric='mse', mode='min')

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self.train_ray),
                resources={'cpu': num_cpu, 'gpu': num_gpu_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric='mse',
                mode='min',
                scheduler=scheduler,
                num_samples=num_samples,
                search_alg=hyperopt_search,   
            ),
            run_config=RunConfig(progress_reporter=reporter),
        )
        results = tuner.fit()
        
        best_result = results.get_best_result('mse', 'min')

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
            best_result.metrics['loss']))
        
        print("Best trial final validation mse: {}".format(
            best_result.metrics['pred mse']))
        
        print("Best trial final validation r2: {}".format(
            best_result.metrics['r2']))
        
        r2_result = results.get_best_result('r2', 'min')
        print("Best trial config: {}".format(r2_result.config))
        print("Best trial final validation loss: {}".format(
            r2_result.metrics['loss']))
        
        print("Best trial final validation mse: {}".format(
            r2_result.metrics['pred mse']))
        
        print("Best trial final validation mse: {}".format(
            r2_result.metrics['r2']))
        
        import pdb; pdb.set_trace()

    
        
            
                