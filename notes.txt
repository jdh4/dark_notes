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
