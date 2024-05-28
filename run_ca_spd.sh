gpu=0
#data_name='distance/16to63_64to71_72to83_ca_new'
data_name='distance/16to63_64to71_72to83_ca'
base_gnn='none_e' # spe
#base_gnn='none' # signnet or naive

for seed in 41 42 43
do
# general runs
#python main.py --pe svd --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn none --seed $seed --dataname $data_name
#python main.py --pe svd --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn none --seed $seed --dataname $data_name --pe_encoder
#python main.py --pe svd --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn $base_gnn --seed $seed --dataname $data_name
#python main.py --pe maglap --pe_dim 32 --q_dim 1 --q 0.1 --gpu_id $gpu --batch_size 512 --base_gnn $base_gnn --seed $seed --dataname $data_name
#python main.py --pe maglap --pe_dim 32 --q_dim 5 --gpu_id $gpu --batch_size 512 --q 0.1 --base_gnn $base_gnn --seed $seed --dataname $data_name
python main.py --pe maglap --pe_dim 32 --q_dim 10 --gpu_id $gpu --batch_size 512 --q 0.05 --base_gnn $base_gnn --seed $seed --dataname $data_name
#python main.py --pe lap --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn $base_gnn --seed $seed --dataname $data_name 


done


