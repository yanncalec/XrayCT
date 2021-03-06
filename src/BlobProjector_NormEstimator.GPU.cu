/*  -*- C -*-  */
/* C interface for BlobProjector_NormEstimator Kernel */
/* This file must be named .cu to be successfully compiled by nvcc */

#include <cuda.h>
#include <stdio.h>
#include "GPUTypes.hpp"
#include "BlobProjector_NormEstimator.Kernel.cu"

cudaError_t BlobProjector_NormEstimator_GPU(BlobType blobtype,  /* The profile function identifier */
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
					    const double *d_T, size_t dimT, bool stripint, /* if stripint = true, d_T is treated as a strip-integral table, otherwise as direct footprint table. */
					    double thetax, double thetay, /* Blob grid's generating vector, the corresponding angle : -pi/2 <= theta < 0*/
					    double splStep, /* sampling step of blob grid */
					    int row, int col, /* Blob grid dimension, even X even */   
					    const int *d_Idx,
					    int nbNode,
					    double lpnorm,
					    bool rownorm,
					    double *d_Y)
{
  /* Invoke Kernel */
  size_t GRIDSIZE = (size_t)ceil(sqrt(nbNode));
  dim3 gridDim(GRIDSIZE, GRIDSIZE);

  BlobProjector_NormEstimator_Kernel<<<gridDim, nbProj>>>(blobtype,
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
							  d_T, dimT, stripint,
							  thetax, thetay,
							  splStep,
							  row, col,
							  d_Idx,
							  nbNode,
							  lpnorm,
							  rownorm,
							  d_Y);

  return cudaGetLastError();
}
