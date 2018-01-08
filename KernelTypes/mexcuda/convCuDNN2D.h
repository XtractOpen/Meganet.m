typedef float NUM_TYPE;

const int NUM_ELEMENTS_SESSION = 7;

enum Operation { Conv, dYdK_T,dYdX_T};



void cudaFreeAll(void** descriptorsAndMem);

char* createConvolutionDescriptors(void** descriptorsAndMem);

char* setConvolutionDescriptors(Operation OP,int* IM_size,int* K_size,int* IM_size_out,void** descriptorsAndMem,int stride);

char* updateAllocatedWorkspace(void** descriptorsAndMem, size_t* nBytesOld, size_t* nBytesNew);

/*
	function performs a full NN forward convolution.
	input: 
		 - Y: is a pointer to populate the result. It is assumed to be already allocated with the right size (also see setup). 
	output:
		 - 	out param is an error massage string. If NULL, it is success. If not NULL - it contains a (hopefully informative) error 
			message that was raised by a cuDNN function.
*/

char* performConvolution(NUM_TYPE const *X, NUM_TYPE const *K, NUM_TYPE* Y, void** descriptorsAndMem, size_t* nBytesAllocated);

char* performConvolutiondYdK(NUM_TYPE const *X, NUM_TYPE const *dY, NUM_TYPE* dK, void** descriptorsAndMem, size_t* nBytesAllocated);

char* performConvolutiondYdX(NUM_TYPE const *K, NUM_TYPE const *dY, NUM_TYPE* dX, void** descriptorsAndMem, size_t* nBytes);