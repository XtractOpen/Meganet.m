#include "mex.h"
#include "gpu/mxGPUArray.h"
//#include "/Users/lruthot/Downloads/cuda/include/cudnn.h"
#include "cudnn.h"
#include "convCuDNN2D.h"
#include<time.h>

/*
 * Host code
 * Y = convCouple(X,X_size,K,K_size,OP)
 OP : 0 for Conv, 1 for dYdK_T, 2 for dYdX_T
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const	*gpuArrIn1;
    mxGPUArray const 	*gpuArrIn2;
	mxGPUArray 			*gpuArrOut = NULL;
    NUM_TYPE 	 		*ArrOut; // this is used for Y in Conv.
    NUM_TYPE const	 	*ArrIn1; // this is used for X in Conv.
	NUM_TYPE const 		*ArrIn2; // this is used as K in Conv.	
	
    int* 				X_size;
	int* 				K_size;
	int					stride;
	int 				ii;
    char const * const 	errId = "parallel:gpu:mex:InvalidInput";
	mwSize 				dims[4];
	const mwSize dimsBytes[] ={1};
	int 				Y_size_out[4];
	char* 				err = NULL;
	void*				descriptorsAndMemNew[NUM_ELEMENTS_SESSION];
	void**				descriptorsAndMem;
	size_t				nBytes;
	size_t*				pnBytes;
	
	Operation 			OP = Conv;
	
	int 				sessionExistsOutside;
    
	
	// clock_t ticks1, ticks2;

	
	
	
    /* Initialize the MathWorks GPU API. */
	// ticks1=clock();
	mxInitGPU();
	
    
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs != 6)&&(nrhs!=7)) {
        mexErrMsgIdAndTxt(errId, "Number of arguments must be 6 or 7");
    }
	
	if (!(mxIsGPUArray(prhs[0]))||!(mxIsGPUArray(prhs[2]))) {
        mexErrMsgIdAndTxt(errId, "Input Kernel/Images are not GPU arrays");
    }
	
	sessionExistsOutside = nrhs == 7;
	
	X_size  	= (int*)mxGetData(prhs[4]);
	OP 			= (Operation)*X_size;
	
	X_size 		= (int*) mxGetData(prhs[1]);
    K_size  	= (int*) mxGetData(prhs[3]);
	
	stride      = *((int*) mxGetData(prhs[5]));
	
	
    
	 /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
	gpuArrIn1 = mxGPUCreateFromMxArray(prhs[0]);
	gpuArrIn2 = mxGPUCreateFromMxArray(prhs[2]);
	
	
    /*
     * Verify that A really is a single array before extracting the pointer.
     */
    if (mxGPUGetClassID(gpuArrIn1) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, "Arrays are not in single precision");
    }
	
	ArrIn1 	= (NUM_TYPE const *)(mxGPUGetDataReadOnly(gpuArrIn1));
	ArrIn2 	= (NUM_TYPE const *)(mxGPUGetDataReadOnly(gpuArrIn2));
	
	
	if (!sessionExistsOutside){
		descriptorsAndMem = descriptorsAndMemNew;
		for (ii = 0; ii < NUM_ELEMENTS_SESSION; ++ii){
			descriptorsAndMem[ii] = NULL;
		}
		err = createConvolutionDescriptors(descriptorsAndMem);
		if (err!=NULL){
			mxGPUDestroyGPUArray(gpuArrIn1);
			mxGPUDestroyGPUArray(gpuArrIn2);
			cudaFreeAll(descriptorsAndMem);
			mexErrMsgIdAndTxt(errId, err);
		}
		pnBytes 		  =	&nBytes;
	}else{
		descriptorsAndMem = (void**)mxGetData(prhs[6]);
		pnBytes			  = (size_t*) descriptorsAndMem[6];
		// mexPrintf("NUM Bytes is: %d\n",*pnBytes);
	}
	
	
	// OP == Conv: 		gpuArrIn1 = X ; gpuArrIn2 = K; gpuArrOut = Y;
	// OP == dYdK_T:	gpuArrIn1 = X ; gpuArrIn2 = dY;  gpuArrOut = dK;
	// OP == dYdX_T:	gpuArrIn1 = K ; gpuArrIn2 = dY;  gpuArrOut = dX;
	
	
	
	err = setConvolutionDescriptors(OP,X_size, K_size, Y_size_out,descriptorsAndMem,stride);
	if (err!=NULL){
		mxGPUDestroyGPUArray(gpuArrIn1);
		mxGPUDestroyGPUArray(gpuArrIn2);
		cudaFreeAll(descriptorsAndMem);
		mexErrMsgIdAndTxt(errId, err);
	}
	
	// ticks1=clock();
	if (OP==Conv){
		for (ii = 0; ii < 4; ++ii){
			dims[ii] = (mwSize)Y_size_out[ii];
		}
		// gpuArrOut = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(gpuArrIn1),dims,
		gpuArrOut = mxGPUCreateGPUArray(4,dims,
                            mxGPUGetClassID(gpuArrIn1),
                            mxGPUGetComplexity(gpuArrIn1),
                            MX_GPU_DO_NOT_INITIALIZE);
		ArrOut = (NUM_TYPE *)(mxGPUGetData(gpuArrOut));
	
		err = performConvolution(ArrIn1,ArrIn2,ArrOut,descriptorsAndMem,pnBytes);
		plhs[0] = mxGPUCreateMxArrayOnGPU(gpuArrOut);	
	
	}else if (OP == dYdK_T){
		for (ii = 0; ii < 4; ++ii){
			dims[ii] = (mwSize)K_size[ii];
		}
		// gpuArrOut = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(gpuArrIn1),dims,
		gpuArrOut = mxGPUCreateGPUArray(4,dims,
                            mxGPUGetClassID(gpuArrIn1),
                            mxGPUGetComplexity(gpuArrIn1),
                            MX_GPU_DO_NOT_INITIALIZE);
		ArrOut = (NUM_TYPE *)(mxGPUGetData(gpuArrOut));
	
		err = performConvolutiondYdK(ArrIn1,ArrIn2,ArrOut,descriptorsAndMem,pnBytes);
		plhs[0] = mxGPUCreateMxArrayOnGPU(gpuArrOut);
	}else if (OP == dYdX_T){
		for (ii = 0; ii < 4; ++ii){
			dims[ii] = (mwSize)X_size[ii];
		}
		// gpuArrOut = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(gpuArrIn2),dims,
		gpuArrOut = mxGPUCreateGPUArray(4,dims,
                            mxGPUGetClassID(gpuArrIn2),
                            mxGPUGetComplexity(gpuArrIn2),
                            MX_GPU_DO_NOT_INITIALIZE);
		ArrOut = (NUM_TYPE *)(mxGPUGetData(gpuArrOut));
	
		err = performConvolutiondYdX(ArrIn1,ArrIn2,ArrOut,descriptorsAndMem,pnBytes);
		plhs[0] = mxGPUCreateMxArrayOnGPU(gpuArrOut);
	}
	
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
	
    mxGPUDestroyGPUArray(gpuArrOut);
	mxGPUDestroyGPUArray(gpuArrIn1);
	mxGPUDestroyGPUArray(gpuArrIn2);
	// cudaDeviceSynchronize();
	// ticks2=clock();
	// mexPrintf("Operation took: %lf, seconds\n",(double)(ticks2-ticks1)/(double)CLOCKS_PER_SEC);
	
	if (!sessionExistsOutside){
		cudaFreeAll(descriptorsAndMem);
	}else{
		cudaDeviceSynchronize();
	}
	if (err != NULL){
		mexErrMsgIdAndTxt(errId, err);	
	}
}