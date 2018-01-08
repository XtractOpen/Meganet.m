#include "cudnn.h"
#include "mex.h"
#include "convCuDNN2D.h"

char* createConvolutionDescriptors(void** descriptorsAndMem){
	char* 							errMsg = NULL;
	size_t*							nBytes;
	cudnnHandle_t 					hCudNN = NULL;
	cudnnTensorDescriptor_t 		pXTensorDesc = NULL;
	cudnnFilterDescriptor_t 		pKFilterDesc = NULL;
	cudnnConvolutionDescriptor_t 	pConvDesc = NULL;
	cudnnTensorDescriptor_t 		pYTensorDesc = NULL;
	cudnnStatus_t 					status;

	status = cudnnCreate(&hCudNN);
	descriptorsAndMem[0] = hCudNN;
	if (status != CUDNN_STATUS_SUCCESS){
		mexPrintf("cudnn creat failed with error %d",status);
		errMsg = "cudnnCreate failed";
		return errMsg;
	}

	status = cudnnCreateTensorDescriptor(&pXTensorDesc);
	descriptorsAndMem[1] = pXTensorDesc;
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnCreateTensorDescriptor for input failed";
		return errMsg;
	}

	status = cudnnCreateFilterDescriptor(&pKFilterDesc);
	descriptorsAndMem[2] = pKFilterDesc;
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnCreateFilterDescriptor failed";
		return errMsg;
	}

	status = cudnnCreateConvolutionDescriptor(&pConvDesc);
	descriptorsAndMem[3] = pConvDesc;
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnCreateConvolutionDescriptor failed";
		return errMsg;
	}
	

	status = cudnnCreateTensorDescriptor(&pYTensorDesc);
	descriptorsAndMem[4] = pYTensorDesc;
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnCreateTensorDescriptor for output failed";
		return errMsg;
	}
	nBytes = (size_t*)malloc(sizeof(size_t));
	// mexPrintf("nBytes allocated\n");
	*nBytes = 0;
	// mexPrintf("nBytes accessed\n");
	descriptorsAndMem[6] = (void*)nBytes;
	return NULL; // function successfully computed the size of the output.
}



char* setConvolutionDescriptors(Operation OP,int* IM_size,int* K_size,int* IM_size_out,void** descriptorsAndMem,int stride){
	char* 							errMsg = NULL;
	//cudnnHandle_t 					hCudNN = NULL;
	cudnnTensorDescriptor_t 		pXTensorDesc = NULL;
	cudnnFilterDescriptor_t 		pKFilterDesc = NULL;
	cudnnConvolutionDescriptor_t 	pConvDesc = NULL;
	cudnnTensorDescriptor_t 		pYTensorDesc = NULL;
	cudnnStatus_t 					status;
		
	// IM_size: We assume a NHWC format
	int n_in = IM_size[3]; // Number of images - originally 128
	int c_in = IM_size[2]; // Number of feature maps per image 
	int h_in = IM_size[1]; // Height of each image
	int w_in = IM_size[0]; // Width of each image  
	int cout_pFilter_in = K_size[3]; // Number of output feature maps  
	int cin_pFilter_in = c_in; // Number of input feature maps, should also equal K_size[2]
	int h_pFilter_in = K_size[1]; // Height of each pFilter
	int w_pFilter_in = K_size[0]; // Width of each pFilter
	int n_out = 0; // Number of output images.
	int c_out = 0; // Number of output feature maps per image.
	int h_out = 0; // Height of each output feature map.
	int w_out = 0; // Width of each output feature map.
	
	int h_pad = div(h_pFilter_in-1,2).quot;
	int w_pad = div(w_pFilter_in-1,2).quot; 

	/* to change to double, chance CUDNN_DATA_FLOAT to CUDNN_DATA_DOUBLE and change each float to double below */

	cudnnDataType_t 				dataType 	  	= CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t 			tensorFormat  	= CUDNN_TENSOR_NCHW; //CUDNN_TENSOR_NHWC CUDNN_TENSOR_NCHW	
	cudnnConvolutionMode_t      	convMode	  	= CUDNN_CROSS_CORRELATION;// can also be CUDNN_CONVOLUTION or CUDNN_CROSS_CORRELATION

	if (K_size[2] != c_in){
		errMsg = "number of input channels in filter is different than the number of input channels in X";
		return errMsg;
	}
	
	// hCudNN 			= (cudnnHandle_t)descriptorsAndMem[0];
	pXTensorDesc 	= (cudnnTensorDescriptor_t)descriptorsAndMem[1];
	pKFilterDesc	= (cudnnFilterDescriptor_t)descriptorsAndMem[2];
	pConvDesc		= (cudnnConvolutionDescriptor_t)descriptorsAndMem[3];
	pYTensorDesc  	= (cudnnTensorDescriptor_t)descriptorsAndMem[4];
	
	
//---------------------------------------
// Set (input) decriptors
//---------------------------------------
	status = cudnnSetTensor4dDescriptor(pXTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnSetTensor4dDescriptor failed";
		if (status == CUDNN_STATUS_BAD_PARAM){
			mexPrintf("Images tensor parameters are (w,h,cin,N): %d,%d,%d,%d\n:",w_in,h_in,c_in,n_in);
			errMsg = "cudnnSetTensor4dDescriptor failed with CUDNN_STATUS_BAD_PARAM";
		}
		return errMsg;;
	}
	
	status = cudnnSetFilter4dDescriptor(pKFilterDesc, dataType,tensorFormat, cout_pFilter_in, cin_pFilter_in, h_pFilter_in, w_pFilter_in);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnSetFilter4dDescriptor failed";
		if (status == CUDNN_STATUS_BAD_PARAM){
			mexPrintf("Filter parameters are (w,h,cin,cout): %d,%d,%d,%d\n:",w_pFilter_in,h_pFilter_in,cin_pFilter_in,cout_pFilter_in);
			errMsg = "cudnnSetFilter4dDescriptor failed with CUDNN_STATUS_BAD_PARAM";
		}
		return errMsg;
	}

	// h_pad and w_pad assume "dirichlet 0 BC".
	status = cudnnSetConvolution2dDescriptor(pConvDesc, h_pad, w_pad, stride, stride, 1, 1, convMode,dataType);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnSetConvolution2dDescriptor failed";
		return errMsg;
	}
	
//------------------------------------------------------------------------------
// Query output tensor, set output tensor descriptor, and allocate output layout
//------------------------------------------------------------------------------
	
	status = cudnnGetConvolution2dForwardOutputDim(pConvDesc,pXTensorDesc,pKFilterDesc, &n_out, &c_out, &h_out, &w_out);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnGetConvolution2dForwardOutputDim failed";
		return errMsg;
	}
	IM_size_out[0] = w_out;IM_size_out[1] = h_out;IM_size_out[2] = c_out;IM_size_out[3] = n_out;
	
	status = cudnnSetTensor4dDescriptor(pYTensorDesc, tensorFormat, dataType, IM_size_out[3], IM_size_out[2], IM_size_out[1], IM_size_out[0]);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnSetTensor4dDescriptor failed";
		return errMsg;
	}
	
	return NULL; // function successfully computed the size of the output.
}


char* updateAllocatedWorkspace(void** descriptorsAndMem, size_t* nBytesOld, size_t* nBytesNew){
	NUM_TYPE* 		pWorkspace	   = NULL;
	char* 	  		errMsg 		   = NULL;
	cudaError_t 	err;
    if ((*nBytesNew > *nBytesOld) || (descriptorsAndMem[5]==NULL)){
		if (descriptorsAndMem[5]!=NULL) 
			cudaFree(descriptorsAndMem[5]);
		err = cudaMalloc((void**)&pWorkspace, *nBytesNew);
		descriptorsAndMem[5] = pWorkspace;
		*nBytesOld = *nBytesNew;
		if (err != cudaSuccess){
			errMsg = "cudaMalloc failed to allocate the workspace memory.";
			return errMsg;
		}
	}
	return NULL;
}


/*
	function performs a full NN forward convolution.
	input: 
		 - Y: is a pointer to populate the result. It is assumed to be already allocated with the right size (also see setup). 
	output:
		 - 	out param is an error massage string. If NULL, it is success. If not NULL - it contains a (hopefully informative) error 
			message that was raised by a cuDNN function.
*/

char* performConvolution(NUM_TYPE const *X, NUM_TYPE const *K, NUM_TYPE* Y, void** descriptorsAndMem, size_t* nBytesAllocated){
				
	char* 							errMsg = NULL;
	cudnnHandle_t 					hCudNN = NULL;
	cudnnTensorDescriptor_t 		pXTensorDesc = NULL;
	cudnnFilterDescriptor_t 		pKFilterDesc = NULL;
	cudnnConvolutionDescriptor_t 	pConvDesc = NULL;
	cudnnTensorDescriptor_t 		pYTensorDesc = NULL;
	cudnnStatus_t 					status;
	
	size_t							nBytesNeeded;
	/* to change to double, chance CUDNN_DATA_FLOAT to CUDNN_DATA_DOUBLE and change each float to double below */
	cudnnConvolutionFwdAlgo_t 	convAlgo      = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;//CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	
	NUM_TYPE*					pWorkspace	   = NULL;
	NUM_TYPE 					alpha 		   = 1.0;
	NUM_TYPE 					beta 		   = 0.0;

	hCudNN 			= (cudnnHandle_t)descriptorsAndMem[0];
	pXTensorDesc 	= (cudnnTensorDescriptor_t)descriptorsAndMem[1];
	pKFilterDesc	= (cudnnFilterDescriptor_t)descriptorsAndMem[2];
	pConvDesc		= (cudnnConvolutionDescriptor_t)descriptorsAndMem[3];
	pYTensorDesc  	= (cudnnTensorDescriptor_t)descriptorsAndMem[4];
	
	
	status = cudnnGetConvolutionForwardWorkspaceSize(hCudNN, pXTensorDesc, pKFilterDesc, pConvDesc, pYTensorDesc, convAlgo, &nBytesNeeded);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnGetConvolutionForwardWorkspaceSize failed";
		if (status == CUDNN_STATUS_BAD_PARAM){
			errMsg = "cudnnGetConvolutionForwardWorkspaceSize failed with CUDNN_STATUS_BAD_PARAM";
		}
		if (status == CUDNN_STATUS_NOT_SUPPORTED){
			errMsg = "cudnnGetConvolutionForwardWorkspaceSize failed with CUDNN_STATUS_NOT_SUPPORTED";
		}
		return errMsg;
	}
	
	errMsg = updateAllocatedWorkspace(descriptorsAndMem,nBytesAllocated,&nBytesNeeded);
	if (errMsg!= NULL)
		return errMsg;
	pWorkspace = (NUM_TYPE*)descriptorsAndMem[5];
//---------------------------------------
// Launch convolution on GPU
//---------------------------------------
 	status = cudnnConvolutionForward(hCudNN, &alpha, pXTensorDesc, X, pKFilterDesc, K, pConvDesc, convAlgo, pWorkspace, *nBytesAllocated, &beta, pYTensorDesc, Y);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnConvolutionForward failed";
		if (status == CUDNN_STATUS_BAD_PARAM){
			errMsg = "cudnnConvolutionForward failed with CUDNN_STATUS_BAD_PARAM";
		}
		return errMsg;
	}
	return NULL;
} 


char* performConvolutiondYdK(NUM_TYPE const *X, NUM_TYPE const *dY, NUM_TYPE* dK, void** descriptorsAndMem, size_t* nBytesAllocated){	
	char* 							errMsg = NULL;
	cudnnHandle_t 					hCudNN = NULL;
	cudnnTensorDescriptor_t 		pXTensorDesc = NULL;
	cudnnFilterDescriptor_t 		pKFilterDesc = NULL;
	cudnnConvolutionDescriptor_t 	pConvDesc = NULL;
	cudnnTensorDescriptor_t 		pYTensorDesc = NULL;
	cudnnStatus_t 					status;
	
	size_t							nBytesNeeded;

	/* to change to double, chance CUDNN_DATA_FLOAT to CUDNN_DATA_DOUBLE and change each float to double below */
	cudnnConvolutionBwdFilterAlgo_t dYdKBwdAlgo 	= CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
	
	NUM_TYPE*					pWorkspace	   = NULL;
	NUM_TYPE 					alpha 		   = 1.0;
	NUM_TYPE 					beta 		   = 0.0;
	

	hCudNN 			= (cudnnHandle_t)descriptorsAndMem[0];
	pXTensorDesc 	= (cudnnTensorDescriptor_t)descriptorsAndMem[1];
	pKFilterDesc	= (cudnnFilterDescriptor_t)descriptorsAndMem[2];
	pConvDesc		= (cudnnConvolutionDescriptor_t)descriptorsAndMem[3];
	pYTensorDesc  	= (cudnnTensorDescriptor_t)descriptorsAndMem[4];
	pWorkspace		= (NUM_TYPE*)descriptorsAndMem[5];
	
	status = cudnnGetConvolutionBackwardFilterWorkspaceSize(hCudNN,pXTensorDesc,pYTensorDesc,pConvDesc,pKFilterDesc,dYdKBwdAlgo,&nBytesNeeded);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnGetConvolutionBackwardFilterWorkspaceSize failed";
		if (status == CUDNN_STATUS_BAD_PARAM){
			errMsg = "cudnnGetConvolutionBackwardFilterWorkspaceSize failed with CUDNN_STATUS_BAD_PARAM";
		}
		if (status == CUDNN_STATUS_NOT_SUPPORTED){
			errMsg = "cudnnGetConvolutionBackwardFilterWorkspaceSize failed with CUDNN_STATUS_NOT_SUPPORTED";
		}
		return errMsg;
	}
	errMsg = updateAllocatedWorkspace(descriptorsAndMem,nBytesAllocated,&nBytesNeeded);
	if (errMsg!= NULL)
		return errMsg;
	
	pWorkspace = (NUM_TYPE*)descriptorsAndMem[5];
	
//---------------------------------------
// Launch convolution on GPU
//---------------------------------------
 	status = cudnnConvolutionBackwardFilter(hCudNN, &alpha, pXTensorDesc, X, pYTensorDesc, dY, pConvDesc, dYdKBwdAlgo, pWorkspace, *nBytesAllocated, &beta, pKFilterDesc, dK);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnConvolutionBackwardFilter failed";
		if (status == CUDNN_STATUS_BAD_PARAM){
			errMsg = "cudnnConvolutionBackwardFilter failed with CUDNN_STATUS_BAD_PARAM";
		}
		return errMsg;
	}
	return NULL;
} 


char* performConvolutiondYdX(NUM_TYPE const *K, NUM_TYPE const *dY, NUM_TYPE* dX, void** descriptorsAndMem, size_t* nBytesAllocated){	
	char* 							errMsg = NULL;
	cudnnHandle_t 					hCudNN = NULL;
	cudnnTensorDescriptor_t 		pXTensorDesc = NULL;
	cudnnFilterDescriptor_t 		pKFilterDesc = NULL;
	cudnnConvolutionDescriptor_t 	pConvDesc = NULL;
	cudnnTensorDescriptor_t 		pYTensorDesc = NULL;
	cudnnStatus_t 					status;
	
	size_t							nBytesNeeded;


	/* to change to double, chance CUDNN_DATA_FLOAT to CUDNN_DATA_DOUBLE and change each float to double below */
	cudnnConvolutionBwdDataAlgo_t dYdXBwdAlgo 	= CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
	
	NUM_TYPE*					pWorkspace	   = NULL;
	NUM_TYPE 					alpha 		   = 1.0;
	NUM_TYPE 					beta 		   = 0.0;

	hCudNN 			= (cudnnHandle_t)descriptorsAndMem[0];
	pXTensorDesc 	= (cudnnTensorDescriptor_t)descriptorsAndMem[1];
	pKFilterDesc	= (cudnnFilterDescriptor_t)descriptorsAndMem[2];
	pConvDesc		= (cudnnConvolutionDescriptor_t)descriptorsAndMem[3];
	pYTensorDesc  	= (cudnnTensorDescriptor_t)descriptorsAndMem[4];
	pWorkspace		= (NUM_TYPE*)descriptorsAndMem[5];
	
	status = cudnnGetConvolutionBackwardDataWorkspaceSize(hCudNN,pKFilterDesc,pYTensorDesc,pConvDesc,pXTensorDesc,dYdXBwdAlgo,&nBytesNeeded);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnGetConvolutionBackwardDataWorkspaceSize failed";
		if (status == CUDNN_STATUS_BAD_PARAM){
			errMsg = "cudnnGetConvolutionBackwardDataWorkspaceSize failed with CUDNN_STATUS_BAD_PARAM";
		}
		if (status == CUDNN_STATUS_NOT_SUPPORTED){
			errMsg = "cudnnGetConvolutionBackwardDataWorkspaceSize failed with CUDNN_STATUS_NOT_SUPPORTED";
		}
		return errMsg;
	}
	
	errMsg = updateAllocatedWorkspace(descriptorsAndMem,nBytesAllocated,&nBytesNeeded);
	pWorkspace = (NUM_TYPE*)descriptorsAndMem[5];
	if (errMsg!= NULL)
		return errMsg;
//---------------------------------------
// Launch convolution on GPU
//---------------------------------------
	
 	status = cudnnConvolutionBackwardData(hCudNN, &alpha, pKFilterDesc, K, pYTensorDesc, dY, pConvDesc, dYdXBwdAlgo, pWorkspace, *nBytesAllocated, &beta, pXTensorDesc, dX);
	if (status != CUDNN_STATUS_SUCCESS){
		errMsg = "cudnnConvolutionBackwardData failed";
		if (status == CUDNN_STATUS_BAD_PARAM){
			errMsg = "cudnnConvolutionBackwardData failed with CUDNN_STATUS_BAD_PARAM";
		}
		return errMsg;
	}
	return NULL;
} 
 

void cudaFreeAll(void** descriptorsAndMem){
	int ii;
	if (descriptorsAndMem[0] != NULL)
		cudnnDestroy((cudnnHandle_t)descriptorsAndMem[0]);// hCudNN
	
	if (descriptorsAndMem[1] != NULL)
		cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)descriptorsAndMem[1]); //pXTensorDesc

	if (descriptorsAndMem[2] != NULL)
		cudnnDestroyFilterDescriptor((cudnnFilterDescriptor_t)descriptorsAndMem[2]);//pKFilterDesc

	if (descriptorsAndMem[3] != NULL)
		cudnnDestroyConvolutionDescriptor((cudnnConvolutionDescriptor_t)descriptorsAndMem[3]);//pConvDesc

	if (descriptorsAndMem[4] != NULL)
		cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)descriptorsAndMem[4]);//pYTensorDesc

	if (descriptorsAndMem[5] != NULL)
		cudaFree(descriptorsAndMem[5]); //pWorkspace
	
	if (descriptorsAndMem[6] != NULL)
		free(descriptorsAndMem[6]); //nBytes
	
	for (ii = 0; ii < NUM_ELEMENTS_SESSION; ++ii){
		descriptorsAndMem[ii] = NULL;
	}
}