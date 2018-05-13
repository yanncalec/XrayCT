/*  -*- C -*-  */
/* C interface for PixDrvProjector Kernel */
/* This file must be named .cu to be successfully compiled by nvcc */

#include <cuda.h>
#include <stdio.h>
#include "GPUTypes.hpp"
#include "PixDrvProjector.Kernel.cu"

cudaError_t PixDrvProjector_GPU(bool fanbeam, 
				int nbProj, 
				const int *d_projset, 
				const double *d_pSrc, 
				const double *d_rSrc, 
				const double *d_rDet,   
				double sizeDet, int pixDet, 
				double spObj,
				int row, int col, 
				const double *d_X,
				double *d_Y,
				bool forward)
{
  /* Invoke Kernel */
  size_t GRIDSIZE = (size_t)ceil(sqrt(row * col));
  //printf("%d\n",GRIDSIZE);
  dim3 gridDim(GRIDSIZE, GRIDSIZE);

  //printf("radius = %f\n", radius);
  PixDrvProjector_Kernel<<<gridDim, nbProj>>>(fanbeam,
					      d_projset,
					      d_pSrc,
					      d_rSrc,
					      d_rDet,
					      sizeDet, pixDet,
					      spObj,
					      row, col,
					      d_X,
					      d_Y,
					      forward);
  
  return cudaGetLastError();
}


