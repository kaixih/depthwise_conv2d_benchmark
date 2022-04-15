# depthwise_conv2d_benchmark

## Description
In this benchmark, we have two implementations for depthwise conv2d backprop
w.r.t filter. The difference is the filter format: KCRS or RSCK, where
K=output_channels, C=input_channels, R=filter_height, S=filter_width. The KCRS
is the default filter format for frameworks like PyTorch and Mxnet, while RSCK
is the default for Tensorflow.

## How to Use:
The `make` will generate four executables, two for each format mentioned above:
`depthwise_conv2d_bwd_filter_[kcrs|rsck](_debug).out`. Note, when
`DEBUG_NON_ATOMIC` compiler macro is set, we will use no atomic to update the
output, which is only served for debugging purpose and will lead to wrong
results.

```bash
$ make
$ bash bench.sh depwise_conv2d_bwd_filter_kcrs.out
...
XXX time(ms): 1.128837
...
```

