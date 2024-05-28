import argparse
import importlib
import os

from torch_geometric.loader import DataLoader
import torch

from utils import setup_seed, load_config, merge_dicts
        

def main():
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--general_config', dest = 'general_config', type=str, default ='sample_config', help='Path to the config file.')
    parser.add_argument('--ray_config', dest = 'ray_config', type=str, default = None, help='Path to the config file.')
    parser.add_argument('--pe_config', dest = 'pe_config', type=str, default = None, help='Path to the config file.')
    parser.add_argument('--device', dest = 'device', type = str, default = '0', help = 'when use ray can set multiple devices')
    parser.add_argument('--mode', dest = 'mode', type=str, default ='train_test', help='[train], [test], [train_test]')
    parser.add_argument('--seed', type=int, default=None)
    # these are for raytune
    parser.add_argument('--num_trial', dest = 'num_trial', type = int, default = 100, help = 'number of trials to tune')
    parser.add_argument('--num_cpu', dest = 'num_cpu', type = int, default = 15, help = 'numer of cpu threads')
    parser.add_argument('--num_gpu_per_trial', dest = 'num_gpu_per_trial', type = float, default = 0.5, help = 'number of gpu usage per trial')
    args = parser.parse_args()

    general_config = load_config('./configs/general/'+str(args.general_config)+'.yaml')
    if args.pe_config is not None:
        pe_config = load_config('./configs/pe/'+str(args.pe_config)+'.yaml')
        config = merge_dicts(general_config, pe_config)
    else:
        config = general_config

    # possibly overwrite random seed
    if args.seed is not None:
        config['utils']['seed'] = args.seed

    print(config)
    config['train']['device'] = args.device
    
    # set thread threshold
    torch.set_num_threads(config['utils']['torch_num_threads'])

    # first set up the seeds
    setup_seed(config['utils']['seed'])

    # initialize the runner
    runner_path = 'runner.'+str(config['task']['name'])+'_runner'
    runner_name = config['task']['name'] + 'Runner'
    module = importlib.import_module(runner_path)
    cls = getattr(module, runner_name)
    runner = cls(config)
    
    runner.save_config() # copy the configs in train files
    if args.mode == 'train':
        runner.train()
    elif args.mode == 'test':
        runner.test(load_statedict = True)
    elif args.mode == 'train_test':
        runner.train()
        runner.test(load_statedict = True)
    elif args.mode == 'tune':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['train']['device'])
        tune_config = load_config('./configs/ray/'+str(args.ray_config)+'.yaml')
        runner.raytune(tune_config = tune_config, num_samples = args.num_trial, num_cpu = args.num_cpu, num_gpu_per_trial = args.num_gpu_per_trial)
    elif args.mode == 'get_result':
        seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for seed_idx, seed in enumerate(seed_list):
            setup_seed(seed)
            runner.train()
            runner.test(load_statedict = True, test_num_idx = seed_idx)
if __name__ == '__main__':
    main()