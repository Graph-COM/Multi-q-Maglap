gine_config='gine_maglap_n_5q_spe_001'
bigine_config='bigine_maglap_n_5q_spe_001'
device=0


for seed in 121 122 123 124 125 126 127 128 129 130
do




#python main.py --general_config hls/dsp/$gine_config --pe_config lap10/lap_gnn --seed $seed --device $device
#python main.py --general_config hls/dsp/$bigine_config --pe_config lap10/lap_gnn --seed $seed --device $device
#python main.py --general_config hls/lut/$gine_config --pe_config lap10/lap_gnn --seed $seed --device $device
#python main.py --general_config hls/lut/$bigine_config --pe_config lap10/lap_gnn --seed $seed --device $device


python main.py --general_config hls/dsp/$gine_config --pe_config maglap10/maglap_5q001_n_spe --seed $seed --device $device
python main.py --general_config hls/dsp/$bigine_config --pe_config maglap10/maglap_5q001_n_spe --seed $seed --device $device
python main.py --general_config hls/lut/$gine_config --pe_config maglap10/maglap_5q001_n_spe --seed $seed --device $device
python main.py --general_config hls/lut/$bigine_config --pe_config maglap10/maglap_5q001_n_spe --seed $seed --device $device

done
