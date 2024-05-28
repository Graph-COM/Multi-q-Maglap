from utils.get_mag_lap import get_walk_profile_from_pe
import torch
def test_maglap(dataset, model, dist='spd'):
    for data in dataset:
        pe, Lambda = data.pe, data.Lambda
        pe = model.complex_handler.merge_real_imag(pe)
        pe = pe.transpose(0, 1)
        Lambda = Lambda.unflatten(1, (model.complex_handler.q_dim, -1))
        Lambda = Lambda.squeeze(0)
        spd = get_walk_profile_from_pe(pe, Lambda, data.degree, 9, output=dist)
        assert torch.abs(spd[data.dist_index[0], data.dist_index[1]] - data.y).sum() == 0.