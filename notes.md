Ran code on adroit v100 at 1:30 pm on Friday (6/5). On first run it took 4:19. On second run (where input file is now available)
it required 1:48. On 3rd run it required 1:46. Cluster flag is for 'adroit'. These runs were done with the following:

```
In file _run_graph_net_nv.py:
cutoff=10
total_epochs=20
batch_per_epoch=150


In quijote_gn_nv.py:
batch=32
```

First run time on Tiger is 4:45 and second run was 2:07

Furthermore, Tensor Cores are optimizing GEMMs (generalized (dense) matrix-matrix multiplies) operations, there are restrictions on the dimensions of the matrices in order to effectively optimize such operations:
For A x B where A has size (M, K) and B has size (K, N):
N, M, K should be multiples of 8
GEMMs in fully connected layers:
Batch size, input features, output features should be multiples of 8
GEMMs in RNNs:
Batch size, hidden size, embedding size, and dictionary size should be multiples of 8

```
>>> import h5py
>>> f = h5py.File('halos_0.h5', 'r')
>>> list(f.keys())
['df']
>>> f['df']
<HDF5 group "/df" (4 members)>
>>> f['df'].keys()
<KeysViewHDF5 ['axis0', 'axis1', 'block0_items', 'block0_values']>
>>> f['df']['axis0']
<HDF5 dataset "axis0": shape (8,), type "|S5">
>>> f['df']['axis1']
<HDF5 dataset "axis1": shape (2946947,), type "<i8">
>>> f['df']['block0_items']
<HDF5 dataset "block0_items": shape (8,), type "|S5">
>>> f['df']['block0_values']
<HDF5 dataset "block0_values": shape (2946947, 8), type "<f4">
>>> ds = f['df']['axis0']
>>> ds
<HDF5 dataset "axis0": shape (8,), type "|S5">
>>> ds[::]
array([b'x', b'y', b'z', b'vx', b'vy', b'vz', b'M14', b'delta'],
      dtype='|S5')
>>> ds = f['df']['axis1']
>>> ds.shape
(2946947,)
>>> ds[0]
0
>>> ds[1]
1
>>> ds = f['df']['block0_values']
>>> ds.shape
(2946947, 8)
>>> ds[0]
array([184.12346  , 779.9073   , 774.8043   ,  69.008804 , 169.7808   ,
       272.0725   ,  53.817314 ,   3.5218508], dtype=float32)
```

load xyz
get neighbors using a kdtree

```
ipdb> xyz.shape                                                                                                               
(2946947, 3)
ipdb> neighbors.shape
(2946947,)
ipdb> all_edges.shape
(2, 59785178)
```

Standardize

[NeighborSampler](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html?highlight=neighbor#torch_geometric.data.NeighborSampler)
