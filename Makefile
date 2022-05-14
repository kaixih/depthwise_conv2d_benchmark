KCRS_FILTER=depwise_conv2d_bwd_filter_kcrs
RSCK_FILTER=depwise_conv2d_bwd_filter_rsck
RSCK_FILTER_NEW=depwise_conv2d_bwd_filter_rsck_new
FWD_NCHW=depwise_conv2d_fwd_nchw
BWD_INPUT_NCHW=depwise_conv2d_bwd_input_nchw

NVCC_OPTS=-gencode arch=compute_86,code=sm_86 \
					-gencode 'arch=compute_80,code="sm_80,compute_80"' \
					-gencode 'arch=compute_70,code="sm_70,compute_70"'

all:
	#nvcc ${KCRS_FILTER}.cu -o ${KCRS_FILTER}.out ${NVCC_OPTS}
	#nvcc ${KCRS_FILTER}.cu -o ${KCRS_FILTER}_debug.out ${NVCC_OPTS} -DDEBUG_NON_ATOMIC
	#nvcc ${RSCK_FILTER}.cu -o ${RSCK_FILTER}.out ${NVCC_OPTS}
	#nvcc ${RSCK_FILTER}.cu -o ${RSCK_FILTER}_debug.out ${NVCC_OPTS} -DDEBUG_NON_ATOMIC
	#nvcc ${RSCK_FILTER_NEW}.cu -o ${RSCK_FILTER_NEW}.out ${NVCC_OPTS}
	nvcc ${FWD_NCHW}.cu -o ${FWD_NCHW}_fast.out ${NVCC_OPTS} -DUSE_FAST_INTDIV
	nvcc ${FWD_NCHW}.cu -o ${FWD_NCHW}.out ${NVCC_OPTS}
	nvcc ${BWD_INPUT_NCHW}.cu -o ${BWD_INPUT_NCHW}_fast.out ${NVCC_OPTS} -DUSE_FAST_INTDIV
	nvcc ${BWD_INPUT_NCHW}.cu -o ${BWD_INPUT_NCHW}.out ${NVCC_OPTS}

clean:
	rm -rf *.out
