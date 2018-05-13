/*  -*- C -*-  */
/* C interface for BlobInterpl Kernel */
/* This file must be named .cu to be successfully compiled by nvcc */

#include <cuda.h>
/* #include <cuda_runtime.h> */
/* #include <cuda_runtime_api.h> */

#include "GPUTypes.hpp"
#include "BlobInterpl.Kernel.cu"

cudaError_t BlobInterpl_GPU(BlobType blobtype,
			    LinOpType linoptype,
			    double radius, 
			    double mul, 
			    double dil,
			    double alpha, double beta,
			    double vthetax, double vthetay,
			    double bgrid_splStep,
			    int bgrid_row, int bgrid_col, 
			    const int *bgrid_mask,
			    int bgrid_nbNode,
			    double wthetax, double wthetay,
			    double igrid_splStep, 
			    int igrid_row, int igrid_col,
			    const int *igrid_Idx,
			    int igrid_nbNode,
			    const double *d_X,
			    double *d_Y)
{
  /* Invoke Kernel */
  size_t N = (size_t)ceil(sqrt(igrid_nbNode / BLOBINTERPL_BLOCKSIZE));
  dim3 gridDim(N, N);

  BlobInterpl_Kernel<<<gridDim, BLOBINTERPL_BLOCKSIZE>>>(blobtype,
							 linoptype,
							 radius, mul, dil,
							 alpha, beta,
							 vthetax, vthetay,
							 bgrid_splStep,
							 bgrid_row, bgrid_col,
							 bgrid_mask,
							 bgrid_nbNode,
							 wthetax, wthetay,
							 igrid_splStep,
							 igrid_row, igrid_col,
							 igrid_Idx,
							 igrid_nbNode,
							 d_X,
							 d_Y);
  /* printf("Kernel end!!\n");   */
  return cudaGetLastError();
}
