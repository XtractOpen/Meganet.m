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
 		

	int 				ii;
    char const * const 	errId = "parallel:gpu:mex:InvalidInput";
	char* 				err = NULL;
	void**				descriptorsAndMem;

	
	plhs[0] = mxCreateDoubleMatrix(NUM_ELEMENTS_SESSION, 1, mxREAL);
	
	
    /* Initialize the MathWorks GPU API. */
	mxInitGPU();
	
	descriptorsAndMem = (void**)mxGetData(plhs[0]);
	for (ii = 0; ii < NUM_ELEMENTS_SESSION; ++ii){
		descriptorsAndMem[ii] = NULL;
	}
	
	err = createConvolutionDescriptors(descriptorsAndMem);
	if (err!=NULL){
		cudaFreeAll(descriptorsAndMem);
		mexErrMsgIdAndTxt(errId, err);
	}
}