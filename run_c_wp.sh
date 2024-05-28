gpu=0
data_name='distance/16to63_64to71_72to83_c_wp'
#data_name='distance/16to63_64to71_72to83_ca'
#base_gnn='none_e'
base_gnn='none_e'

for seed in 41 42 43
do
# general runs
#python main.py --pe svd --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn none --seed $seed --dataname $data_name --out_dim 5
#python main.py --pe svd --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn none --seed $seed --dataname $data_name --out_dim 5 --pe_encoder
#python main.py --pe svd --pe_dim 32 --gpu_id $gpu --batch_size 512 --base_gnn $base_gnn --seed $seed --dataname $data_name --out_dim 5
#python main.py --pe maglap --pe_dim 32 --q_dim 1 --q 0.1 --gpu_id $gpu --batch_size 128 --base_gnn $base_gnn --seed $seed --dataname $data_name --out_dim 5
#python main.py --pe maglap --pe_dim 32 --q_dim 1 --q 0.1 --gpu_id $gpu --batch_size 128 --base_gnn $base_gnn --seed $seed --dataname $data_name --out_dim 5 --pe_encoder
python main.py --pe maglap --pe_dim 32 --q_dim 5 --gpu_id $gpu --q 0.1 --base_gnn $base_gnn --seed $seed --dataname $data_name --out_dim 5 --batch_size 128
#python main.py --pe maglap --pe_dim 32 --q_dim 10 --gpu_id $gpu --q 0.05 --base_gnn $base_gnn --seed $seed --dataname $data_name --out_dim 5 --batch_size 64 
#python main.py --pe maglap --pe_dim 32 --q_dim 10 --gpu_id $gpu --q 0.05 --base_gnn $base_gnn --seed $seed --dataname $data_name --out_dim 5 --batch_size 128 --pe_encoder
#python main.py --pe lap --pe_dim 32 --gpu_id $gpu --batch_size 128 --base_gnn $base_gnn --seed $seed --dataname $data_name --out_dim 5
#python main.py --pe lap --pe_dim 32 --gpu_id $gpu --batch_size 128 --base_gnn $base_gnn --seed $seed --dataname $data_name --out_dim 5 --pe_encoder


done


