/*  -*- C -*-  */
/* C interface for BlobMatching Kernel */
/* This file must be named .cu to be successfully compiled by nvcc */

#include <cuda.h>
#include <stdio.h>
#include "GPUTypes.hpp"
#include "BlobMatching.Kernel.cu"

cudaError_t BlobMatching_GPU(BlobType blobtype,  /* The profile function identifier */
			     double radius, 
			     double mul, 
			     double dil, /* radius is that of dilated blob */
			     double alpha, double beta,			      
			     bool fanbeam, 
			     int nbProj, 
			     const int *d_projset, 
			     const double *d_pSrc, 
			     const double *d_rSrc, 
			     const double *d_rDet,   
			     double sizeDet, int pixDet, 
			     const double *d_T, size_t dimT,
			     double thetax, double thetay, /* Blob grid's generating vector, the corresponding angle : -pi/2 <= theta < 0*/
			     double splStep, /* sampling step of blob grid */
			     int row, int col, /* Blob grid dimension, even X even */   
			     const int *d_Idx,
			     int nbNode,
			     const double *d_X,
			     const bool *d_M,
			     const double *d_Y,
			     double *d_Z)
{
  /* Invoke Kernel */
  size_t GRIDSIZE = (size_t)ceil(sqrt(nbNode));
  //printf("%d\n",GRIDSIZE);
  dim3 gridDim(GRIDSIZE, GRIDSIZE);

  BlobMatching_Kernel<<<gridDim, nbProj>>>(blobtype,
					   radius,
					   mul,
					   dil,
					   alpha, beta,
					   fanbeam,
					   d_projset,
					   d_pSrc,
					   d_rSrc,
					   d_rDet,
					   sizeDet, pixDet,
					   d_T, dimT, 
					   thetax, thetay,
					   splStep,
					   row, col,
					   d_Idx,
					   nbNode,
					   d_X,
					   d_M,
					   d_Y,
					   d_Z);

  return cudaGetLastError();
}
