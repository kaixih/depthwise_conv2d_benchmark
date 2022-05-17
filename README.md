# Depthwise Conv2d Benchmark with Fast Int Division

## Description
We benchmark the depthwise conv2d implementations of Tensorflow with the fast
integer division for index computation.

## How to Use:
The `make` will generate four executables, two for each format mentioned above:
`depthwise_conv2d_bwd_filter_[kcrs|rsck](_debug).out`. Note, when
`DEBUG_NON_ATOMIC` compiler macro is set, we will use no atomic to update the
output, which is only served for debugging purpose and will lead to wrong
results.

```bash
The benchmark script will run three convolutions (conv2d forward, conv2d
backprop w.r.t input, and conv2d backprop w.r.t filter) with the fast int
division turned on and off respectively.
$ make
$ bash bench.sh
...
Log is stored in log_xxx.txt
```

