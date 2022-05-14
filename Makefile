FWD_NCHW=depwise_conv2d_fwd_nchw
BWD_INPUT_NCHW=depwise_conv2d_bwd_input_nchw
BWD_FILTER_NCHW=depwise_conv2d_bwd_filter_nchw

NVCC_OPTS=-gencode arch=compute_86,code=sm_86 \
					-gencode 'arch=compute_80,code="sm_80,compute_80"' \
					-gencode 'arch=compute_70,code="sm_70,compute_70"'

all:
	nvcc ${FWD_NCHW}.cu -o ${FWD_NCHW}_fast.out ${NVCC_OPTS} -DUSE_FAST_INTDIV
	nvcc ${FWD_NCHW}.cu -o ${FWD_NCHW}.out ${NVCC_OPTS}
	nvcc ${BWD_INPUT_NCHW}.cu -o ${BWD_INPUT_NCHW}_fast.out ${NVCC_OPTS} -DUSE_FAST_INTDIV
	nvcc ${BWD_INPUT_NCHW}.cu -o ${BWD_INPUT_NCHW}.out ${NVCC_OPTS}
	nvcc ${BWD_FILTER_NCHW}.cu -o ${BWD_FILTER_NCHW}_fast.out ${NVCC_OPTS} -DUSE_FAST_INTDIV
	nvcc ${BWD_FILTER_NCHW}.cu -o ${BWD_FILTER_NCHW}.out ${NVCC_OPTS}

clean:
	rm -rf *.out *.txt
