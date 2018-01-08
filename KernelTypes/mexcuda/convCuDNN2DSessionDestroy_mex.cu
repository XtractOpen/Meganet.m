#include "mex.h"
#include "gpu/mxGPUArray.h"
//#include "/Users/lruthot/Downloads/cuda/include/cudnn.h"
#include "cudnn.h"
#include<time.h>
#include"convCuDNN2D.h"

/*
 * Host code
 * Y = convCouple(X,X_size,K,K_size,OP)
 OP : 0 for Conv, 1 for dYdK_T, 2 for dYdX_T
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    //char const * const 	errId = "parallel:gpu:mex:InvalidInput";
	void**				descriptorsAndMem;
    /* Initialize the MathWorks GPU API. */
	
	mxInitGPU();
	descriptorsAndMem = (void**)mxGetData(prhs[0]);
	cudaFreeAll(descriptorsAndMem);
}