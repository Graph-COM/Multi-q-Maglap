q=0.05
gpu=0
for seed in 41 42 43 44 45
do
# constant q
python main_sort.py --base_gnn transformer --direct bi --pe maglap --q $q --dynamic_q --q_dim 5 --pe_dim 25 --pe_encoder spe --gpu_id $gpu --seed $seed --batch_size 48 --epochs 5
done
