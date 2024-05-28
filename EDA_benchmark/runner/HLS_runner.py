import numpy as np
import wandb
import shutil
import yaml
import importlib
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import r2_score

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

class HLSRunner():
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
        self.init_wandb()
        if tune_parameter_config['criterion'] == 'L1':
            self.criterion = nn.L1Loss()
        elif tune_parameter_config['criterion'] == 'SmoothL1':  
            self.criterion = nn.SmoothL1Loss()
        elif tune_parameter_config['criterion'] == 'MSE':
            self.criterion = nn.MSELoss()
        # initialize the datasets and dataloader
            
        module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
        attribute_name = self.config['task']['name']+'DataProcessor'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.train_data = cls(self.config, 'train')
        self.valid_data = cls(self.config, 'valid')
        batch_size = tune_parameter_config['batch_size']
        self.train_loader = DataLoader(self.train_data, batch_size = batch_size, shuffle = True)
        self.val_loader = DataLoader(self.valid_data, batch_size = batch_size)

        # define the model, optimizer, and scheduler here
        module_path = 'models.'+self.config['task']['name']+'.'+self.config['task']['name']+'_model'
        attribute_name = self.config['task']['name']+'Model'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.model = cls(tune_parameter_config, self.config['task']['target'])
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

        for epoch_idx in tqdm(range(self.config['train']['epoch'])):
            train_loss, train_r2 = self.train_one_epoch(self.train_loader, 'train', epoch_idx)
            valid_loss, valid_r2 = self.train_one_epoch(self.val_loader, 'valid', epoch_idx)
            self.scheduler.step()
            self.save_model(valid_loss, epoch_idx)
            train.report({'loss' : valid_loss, 'mse': valid_loss, 'r2' : valid_r2})
    
    def train(self):
        self.init_wandb()
        # initialize the datasets and dataloader
        module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
        attribute_name = self.config['task']['name']+'DataProcessor'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.train_data = cls(self.config, 'train')
        self.valid_data = cls(self.config, 'valid')
        batch_size = self.config['train']['batch_size']
        self.train_loader = DataLoader(self.train_data, batch_size = batch_size, shuffle = True)
        self.val_loader = DataLoader(self.valid_data, batch_size = batch_size)
        
        # define the model, optimizer, and scheduler here
        module_path = 'models.'+self.config['task']['name']+'.'+self.config['task']['name']+'_model'
        attribute_name = self.config['task']['name']+'Model'
        module = importlib.import_module(module_path)
        
        cls = getattr(module, attribute_name)
        self.model = cls(self.config['model'], self.config['task']['target'])
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
        self.mape.to(self.device)

        # initialize the normalization of regression
        self.min_target = 10000
        self.max_target = 0
        for data in self.train_data:
            target = getattr(data, self.config['task']['target'], None)
            if target > self.max_target:
                self.max_target = target
            if target < self.min_target:
                self.min_target = target
        self.min_target = self.min_target.to(self.device)
        self.max_target = self.max_target.to(self.device)

        for epoch_idx in tqdm(range(self.config['train']['epoch'])):
            train_loss, train_r2 = self.train_one_epoch(self.train_loader, 'train', epoch_idx)
            valid_loss, valid_r2 = self.train_one_epoch(self.val_loader, 'valid', epoch_idx)
            self.scheduler.step()
            self.save_model(valid_loss, epoch_idx)

    def train_one_epoch(self, data_loader, mode, epoch_idx):
        epoch_loss = 0
        total_samples = 0
        pred_list = []
        y_list = []
        for batch_data in data_loader:
            batch_data.to(self.device)
            edge_index = batch_data.edge_index
            edge_attr = batch_data.edge_attr
            if self.config['train']['directed'] == 0:
                edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce = 'add')
                batch_data.edge_index = edge_index
                batch_data.edge_attr = edge_attr
            y = getattr(batch_data, self.config['task']['target'], None)
            if mode == 'train':
                self.model.train()
                self.optimizer.zero_grad()
                if self.config['model']['name'] in ['GPS','GPSSE']:
                    self.model.middle_model.redraw_projection.redraw_projections()
                    pred = self.model(batch_data)
                else:
                    pred = self.model(batch_data)
            elif mode == 'valid':
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(batch_data)
            pred_list.append(pred.detach().cpu().numpy().reshape(-1))
            y_list.append(y.detach().cpu().numpy().reshape(-1))
            batch_loss = self.criterion(pred, y.reshape(-1, 1))
            if mode == 'train':
                batch_loss.backward()
                self.optimizer.step()
            epoch_loss = epoch_loss + batch_loss.item() * (torch.max(batch_data.batch).item() + 1)
            total_samples = total_samples + (torch.max(batch_data.batch).item() + 1)
        epoch_loss = epoch_loss / total_samples
        pred_list = np.concatenate(pred_list)
        y_list = np.concatenate(y_list)
        epoch_r2 = r2_score(pred_list, y_list)
        self.write_log({'loss': epoch_loss, 'r2': epoch_r2}, epoch_idx, mode)
        return epoch_loss, epoch_r2
     
    def test(self, load_statedict = True, test_num_idx = 0):
        if load_statedict:
            self.device = int(self.config['train']['device'])
            module_path = 'models.'+self.config['task']['name']+'.'+self.config['task']['name']+'_model'
            attribute_name = self.config['task']['name']+'Model'
            module = importlib.import_module(module_path)
            cls = getattr(module, attribute_name)
            self.model = cls(self.config['model'], self.config['task']['target'])
            dict_path = find_latest_model(self.train_folder, 'model')
            state_dict = torch.load(dict_path) 
            self.model.load_state_dict(state_dict)
            self.model = self.model.to('cuda:'+str(self.device) if torch.cuda.is_available() else 'cpu')
            self.model.eval()
        else:
            self.model.eval() 
        # start testing
        test_list = ['test', 'test_othertype', 'test_real']
        #test_list = ['test']
        self.test_data_dict = {}
        self.test_loader_dict = {}
        for test_name in test_list:
            module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
            attribute_name = self.config['task']['name']+'DataProcessor'
            module = importlib.import_module(module_path)
            cls = getattr(module, attribute_name)
            test_data = cls(self.config, test_name)
            self.test_data_dict[test_name] = test_data
            test_loader = DataLoader(test_data, batch_size = 1)
            self.test_loader_dict[test_name] = test_loader
        table = PrettyTable(['test set name', '# of samples', 'mse loss', 'r2 score'])
        for test_name in test_list:
            mse, r2 = self.test_a_task(test_name)
            row = [str(test_name), str(len(self.test_data_dict[test_name])), mse, r2]
            table.add_row(row)
            if self.config['train']['wandb'] == 1:
                wandb.run.summary[test_name + '-mse'] = mse
                wandb.run.summary[test_name + '-r2'] = r2

        if test_num_idx == 0:
            with open(self.result_csv, 'w', newline='') as f_output:
                f_output.write(table.get_csv_string())
        else:
            with open(self.result_csv, 'a', newline='') as f_output:
                f_output.write(table.get_csv_string())
        print(table)
        
    def test_a_task(self, testset_name):
        pred_list = []
        label_list = []
        for data_idx, batch_data in tqdm(enumerate(self.test_loader_dict[testset_name])):
            batch_data.to(self.device)
            edge_index = batch_data.edge_index
            edge_attr = batch_data.edge_attr
            if self.config['train']['directed'] == 0:
                edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce = 'add')
                batch_data.edge_index = edge_index
                batch_data.edge_attr = edge_attr
            y = getattr(batch_data, self.config['task']['target'], None)
            if data_idx == 55 and testset_name == 'test_real':
                continue
            with torch.no_grad():
                if self.config['model']['name'] in ['GPS','GPSSE']:
                    self.model.middle_model.redraw_projection.redraw_projections()
                    pred = self.model(batch_data)
                else:
                    pred = self.model(batch_data)
            label_list.append(y.reshape(-1, 1))
            pred_list.append(pred)
        label_all = torch.stack(label_list, 0).squeeze(1)
        label_all_formape = torch.clamp(label_all, min = 1)
        pred_all = torch.stack(pred_list, 0).squeeze(1)
        mse = self.mse(pred_all.cpu(), label_all.cpu()).item()
        r2 = r2_score(pred_all.cpu(), label_all.cpu())
        return mse, r2

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
                wandb.log({mode+ ' ' + str(key): items[key]}, step = epoch_idx)
    def init_wandb(self):
        if self.config['train']['wandb'] == 1:
            #wandb.init(project='EDA_benchmark', name = self.config['task']['name']+'_'+str(self.config['task']['type'])+'_'+self.config['task']['target']+'_'+self.config['model']['name'])
            wandb.login(key="cbbd9da073d5e6615442b343b4436e3b723f16da")
            wandb.init(project='EDA_benchmark_HLS', config=self.config)
    def save_model(self, valid_metric, epoch_idx):
        if valid_metric < self.best_valid_metric:
            self.best_valid_metric = valid_metric
            delete_file_with_head(self.train_folder, 'model')
            torch.save(self.model.state_dict(), self.train_folder+'model'+'_epoch'+str(epoch_idx)+'.pth')

    def raytune(self, tune_config, num_samples, num_cpu, num_gpu_per_trial):
        reporter = CLIReporter(parameter_columns=['hidden_dim'],metric_columns=['loss', 'mse', 'r2'])
        # init ray tune
        dropout_p = hp.choice('dropout_p', tune_config['dropout'])
        if self.config['model'].get('pe_file_name') in ['lap_naive', 'maglap_1q_naive'] and self.config['model']['name'] in ['GPS', 'GPSSE']:
            hidden_dim = 14 + 28 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1]))
        else: 
            hidden_dim = 28 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1]))
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
        'attn_kwargs': {'dropout': dropout_p},
        }
        tune_parameter_config = {**self.config['model'], **tune_parameter_config}
        scheduler = ASHAScheduler(
            max_t=800,
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

        print("Best trial final validation mape: {}".format(
            best_result.metrics['mape']))
        
        print("Best trial final validation mse: {}".format(
            best_result.metrics['mse']))
        
        mape_result = results.get_best_result('mape', 'min')
        print("Best trial config: {}".format(mape_result.config))
        print("Best trial final validation loss: {}".format(
            mape_result.metrics['loss']))

        print("Best trial final validation mape: {}".format(
            mape_result.metrics['mape']))
        
        print("Best trial final validation mse: {}".format(
            mape_result.metrics['mse']))
        
        self.test_best_model(best_result)
        import pdb; pdb.set_trace()

    
        
            
                