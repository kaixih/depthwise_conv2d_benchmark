# Depthwise Conv2d Benchmark with Fast Int Division

## Description
We benchmark the depthwise conv2d implementations of Tensorflow with the fast
integer division for index computation.

## How to Use:
The benchmark script will run three convolutions (conv2d forward, conv2d
backprop w.r.t input, and conv2d backprop w.r.t filter) with the fast int
division turned on and off respectively.

```bash
$ make
$ bash bench.sh
...
Log is stored in log_xxx.txt
```

