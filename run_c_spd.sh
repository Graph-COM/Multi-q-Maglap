gpu=0
data_name='distance/16to63_64to71_72to83_c'
#data_name='distance/16to63_64to71_72to83_ca'
base_gnn='none_e'
#base_gnn='none'

for seed in 41 42 43
do
# general runs
#python main.py --pe svd --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn $base_gnn --seed $seed --dataname $data_name 
#python main.py --pe maglap --pe_dim 32 --q_dim 5 --gpu_id $gpu --batch_size 128 --q 0.1 --base_gnn $base_gnn --seed $seed --dataname $data_name --epochs 20
#python main.py --pe maglap --pe_dim 32 --q_dim 10 --gpu_id $gpu --batch_size 512 --q 0.05 --base_gnn $base_gnn --seed $seed --dataname $data_name
python main.py --pe maglap --pe_dim 32 --q_dim 15 --gpu_id $gpu --batch_size 512 --q 0.033 --base_gnn $base_gnn --seed $seed --dataname $data_name
#python main.py --pe maglap --pe_dim 32 --q_dim 1 --q 0.1 --gpu_id $gpu --batch_size 512 --base_gnn $base_gnn --seed $seed --dataname $data_name
#python main.py --pe lap --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn $base_gnn --seed $seed --dataname $data_name

done


