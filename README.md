# What Are Good Positional Encodings for
Directed Graphs?

This is the official implementation for *Multi-q Magnetic Laplacian Positional Encodings* proposed in paper "What Are Good Positional Encodings for
Directed Graphs".

Feel free to contact yhuang903@gatech.edu if there is any question.


## Paper Overview 
Positional encodings (PE) for graphs are essential in constructing powerful and
expressive graph neural networks and graph transformers as they effectively capture
relative spatial relations between nodes. While PEs for undirected graphs have
been extensively studied, those for directed graphs remain largely unexplored,
despite the fundamental role of directed graphs in representing entities with strong
logical dependencies, such as those in program analysis and circuit designs. This
work studies the design of PEs for directed graphs that are expressive to represent
desired directed spatial relations. We first propose walk profile, a generalization of
walk counting sequence to directed graphs. We identify limitations in existing PE
methods—including symmetrized Laplacian PE, Singular Value Decomposition
PE, and Magnetic Laplacian PE—in their ability to express walk profiles. To
address these limitations, we propose the Multi-q Magnetic Laplacian PE, which
extends Magnetic Laplacian PE with multiple potential factors. This simple variant
turns out to be capable of provably expressing walk profiles. Furthermore, we
generalize previous basis-invariant and stable networks to handle complex-domain
PEs decomposed from Magnetic Laplacians. Our numerical experiments demon-
strate the effectiveness of Multi-q Magnetic Laplacian PE with a stable neural
architecture, outperforming previous PE methods (with stable networks) on predict-
ing directed distances/walk profiles, sorting network satisfiability, and on general
circuit benchmarks.


## Code Usage
### Requirement
See ``requirements.txt``

### Distance Prediction

**Dataset generation.** To generate the dataset, step into ``./data_utils`` and run
```
python generate_random_graphs.py --out_path=../data/distance/ --connected --acyclic --n_train=16,64 --n_valid=64,72 --n_test=72,84
```
to generate directed acyclic graphs. To generate regular directed graphs, remove '--acyclic'. By default the node pairs are lalelled with shortest path distance. 
To generate longest path distance or walk profile, set DIST='lpd' or 'wp' at Line 41 of generate_random_graphs.py. 


After finishing dataset generation, run ``run_ca_spd.sh`` to reproduce the shortest path distance results for Multi-q Mag-PE on connected, acyclic directed graphs. 
Others scripts ``run_ca_lpd.sh``, ``run_ca_wp.sh``, ``run_c_spd.sh``,``run_c_lpd.sh``,``run_c_wp.sh`` can reproduce other results correspondingly.


### Sorting Network
**Dataset generation.** To generate the dataset, step into ``./data_utils`` and run
```
python generate_sorting.py
```

After finish dataset generation, run ``run_sort.sh`` to reproduce the result of Multi-q Mag-PE.



### Circuit Property Prediction
The code for this part is in an independent directory ``./EDA_benchmark``. Step into `./EDA_benchmark` as the first step.

To reproduce the results of Open Circuit Benchmark, run ``run_amp.sh``. By default the target is Gain and backbone GNN is BIGINE. To get results of other targets, 
set ``target=bw`` or ``target=pm`` at Line 5 of ``run_amp.sh``. To change backbone GNN to GINE, set ``gine_config=gine_10q001``.


To reproduce the results of High-level synthesis, run ``run_hls.sh``.


 