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

```
ipdb> neighbors.shape  
(2946947,)
                                                                                                                             
ipdb> neighbors[:2]   
array([array([      0, 2607064, 1619670, 2836332, 2088572, 1354055,  776620,
        447072, 1750263, 1704638, 1961557, 2607065,  247825, 2503895,
       1753235, 1226649, 2088570, 2503893, 1412263, 2600271, 1412265,
       1084998, 1961556,  694087,  633029, 1583070, 1473881, 1849466,
       1965250, 2234392,  847565,  327754,   29987, 1104600,   51925,
       1275191, 2320634, 2497618,  918902, 1903957, 2844599, 1412264,
        815228, 1849467, 1086183,  571424, 2320633, 2844598,   37754,
        566503, 1545648,  944466, 1509913, 1801430, 1852629, 2836330,
       2600245,  958315, 1183956, 1473885,  893083,  515857, 2027221,
       2023217,  242630, 1380861, 1380858, 1750262,  138595,  227325,
        786051, 2408815, 2026970,  869525, 1017876,  170547, 2315379,
       2320636,  503654, 1380857, 1903952, 2600269, 2402974, 2163626]),
       array([      1, 2352335, 2647719, 2540974, 1928320, 1364502, 1679841,
       1040662, 2268676,  762382, 1393375, 1987646, 2763915,  898310,
       1928333, 1682478, 1040659,  898307,  623409, 1261308, 1987644,
        302025,  790077, 2448592,  275519, 1773575, 1336553, 1058585,
       2764416,  251617, 2540979,  220685,   10599, 2893995, 2118250,
       2352345,  131547,  415721, 2122706,  410264, 1423423,  315012,
        402271, 2273669, 2190626,  771264, 1819989, 2055127, 2190952,
       1092857])], dtype=object)
```


Standardize

[NeighborSampler](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html?highlight=neighbor#torch_geometric.data.NeighborSampler)

```
ipdb> p graph_data
{'graph': Data(edge_index=[2, 59785178], x=[2946947, 7], y=[2946947, 2]),
'trainloader': <torch_geometric.data.sampler.NeighborSampler object at 0x2b9834bb5f90>,
'column_description': 'x columns are [x, y, z, vx, vy, vz, M]; everything has been
scaled to be std=1. y columns are [bias, mask], where mask=1 indicates that the node should
be used as a receiver for training; mask=0 indicates that the node is too close to the edge.
Multiply the node-wise loss by the mask during training.', 'pos_scale': tensor(288.6028),
'vel_scale': tensor(315.9276), 'M14_scale': tensor(0.3377)}
```

```
ipdb> bin_count=np.bincount(graph_data['graph'].edge_index.flatten())
ipdb> bin_count.mean()           
40.57431504536729
ipdb> bin_count.std()              
21.29062588527445
```

```
ipdb> graph_data['graph'].y[:10] 
tensor([[3.5219, 1.0000],
        [2.1530, 1.0000],
        [2.3769, 1.0000],
        [2.6916, 1.0000],
        [2.9699, 1.0000],
        [2.5331, 0.0000],
        [2.2778, 1.0000],
        [2.5825, 1.0000],
        [2.4690, 1.0000],
        [2.6946, 1.0000]])
        
ipdb> graph_data['graph'].x[:10]  
tensor([[-1.0945e+00,  9.6987e-01,  9.5219e-01,  2.1843e-01,  5.3740e-01,
          8.6119e-01,  1.5938e+02],
        [ 3.8896e-01,  4.9396e-01,  5.0292e-03,  6.6556e-01,  3.3598e-01,
          3.2476e-01,  1.0819e+02],
        [-3.0594e-01, -3.8648e-01,  1.2946e+00,  1.3205e+00, -4.5691e-01,
          5.5381e-01,  1.0466e+02],
        [ 8.8398e-01,  3.6200e-02, -1.5367e+00,  2.2474e-01,  1.8110e-01,
          3.5767e-01,  1.0215e+02],
        [ 6.4622e-01,  1.2295e-01, -2.5183e-01, -8.4634e-02,  1.9930e+00,
          8.2380e-01,  9.6967e+01],
        [ 1.7036e+00,  5.3687e-01,  1.5162e+00,  1.1046e+00,  1.5501e-01,
          3.1554e-01,  9.3363e+01],
        [ 1.6973e+00, -1.0203e+00, -6.8157e-01,  1.1849e+00,  6.1331e-01,
         -1.3412e+00,  8.9502e+01],
        [-4.1678e-01,  1.2448e+00, -1.3983e+00, -2.1441e+00,  6.1721e-02,
          7.4924e-01,  8.5232e+01],
        [-4.1394e-01,  3.5528e-01, -2.9476e-01, -7.3135e-01,  2.6047e-01,
         -8.1957e-01,  8.2431e+01],
        [-9.5565e-01, -8.8100e-01,  1.0726e+00, -9.3461e-01,  8.6351e-01,
          4.5729e-01,  8.1377e+01]])
          
ipdb> graph_data['graph'].edge_index[:10]
tensor([[2607064, 1619670, 2836332,  ...,  484132, 1839981, 1897538],
        [      0,       0,       0,  ..., 2946946, 2946946, 2946946]])
```

```
ipdb> type(ogn)                 
<class 'quijote_gn_nv.GN'>

ipdb> print(ogn)                 
GN(
  (msg_fnc): Sequential(
    (0): Linear(in_features=11, out_features=500, bias=True)
    (1): ReLU()
    (2): Linear(in_features=500, out_features=500, bias=True)
    (3): ReLU()
    (4): Linear(in_features=500, out_features=500, bias=True)
    (5): ReLU()
    (6): Linear(in_features=500, out_features=100, bias=True)
  )
  (node_fnc): Sequential(
    (0): Linear(in_features=104, out_features=500, bias=True)
    (1): ReLU()
    (2): Linear(in_features=500, out_features=500, bias=True)
    (3): ReLU()
    (4): Linear(in_features=500, out_features=500, bias=True)
    (5): ReLU()
    (6): Linear(in_features=500, out_features=1, bias=True)
  )
)
```

GN derives from torch_geometric.nn.MessagePassing

P100 (Monday morning, with file generated from scratch):

```
Total time: 141.284 s
File: /scratch/gpfs/jdh4/gn/generate_halo_data_nv.py
Function: generate_data at line 15

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
@profile
    16                                           def generate_data(realization,cluster):
    ...
55                                           # compute density field of the snapshot (density constrast d = rho/<rho>-1)
    56         1   65608548.0 65608548.0     46.4      delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)
    57         1    1796157.0 1796157.0      1.3      delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    58                                           
    59                                           # # Smooth density field:
    60                                           
    61                                           # smooth the field on a given scale
    62         1   25427061.0 25427061.0     18.0      W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
    63         1   47363841.0 47363841.0     33.5      delta_smoothed = SL.field_smoothing(delta, W_k, threads)
```

```
Total time: 240.662 s
File: /scratch/gpfs/jdh4/gn/quijote_gn_nv.py
Function: load_graph_data at line 32

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    32                                           @profile
    33                                           def load_graph_data(realization=0, cutoff=30):
    34         1          2.0      2.0      0.0      try:
    35         1         94.0     94.0      0.0          cur_data = pd.read_hdf('halos_%d.h5'%(realization,))
    36         1          2.0      2.0      0.0      except:
    37         1      70458.0  70458.0      0.0          from generate_halo_data_nv import generate_data
    38         1  141707954.0 141707954.0     58.9          generate_data(realization, get_cluster())
    39         1     143492.0 143492.0      0.1          cur_data = pd.read_hdf('halos_%d.h5'%(realization,))
    40                                           
    41                                           # # Now, let's connect nearby halos:
    42                                           
    43         1    1963591.0 1963591.0      0.8      xyz = np.array([cur_data.x, cur_data.y, cur_data.z]).T
    44         1    2849210.0 2849210.0      1.2      tree = KDTree(xyz)
    45                                           
    46                                           # ## Let's see what a good radius is. Let's aim for ~8 particles or so for average
    47                                           
    48         1          3.0      3.0      0.0      region_of_influence = cutoff
    49                                           
    50                                               #plt.hist(tree.query_radius(xyz, region_of_influence, count_only=True)-1, bins=31);
    51                                               #plt.xlabel('Number with')
    52                                               #plt.ylabel('Number of neighbors')
    53                                           
    54                                           # ## So, let's create the adjacency matrix:
    55                                           
    56         1   43497352.0 43497352.0     18.1      neighbors = tree.query_radius(xyz, region_of_influence, sort_results=True, return_distance=True)[0]
    ...
```
