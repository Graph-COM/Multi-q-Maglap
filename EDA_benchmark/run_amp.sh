gine_config='bigine_10q001'
#gine_config='gine_10q001'
#seed=123
device=0
target=gain

for seed in 121 122 123 124 125
#for seed in 121
do

#python main.py --general_config amp/$target/$gine_config --pe_config lap10/lap_n_spe_e_spe --seed $seed --device $device
#python main.py --general_config amp/$target/$gine_config --pe_config maglap10/maglap_1q_n_spe_e_spe --seed $seed --device $device

python main.py --general_config amp/$target/$gine_config --pe_config maglap10/maglap_5q001_n_spe_e_spe --seed $seed --device $device

done
