/*  -*- C -*-  */
/* Blob interpolation kernel funcions */

#ifndef _BLOBINTERPL_KERNEL_H_
#define _BLOBINTERPL_KERNEL_H_

#include "GPUDevFunc.cu"
#include "BlobFunc.cu"
#include "GPUTypes.hpp"

/* Blob represented function : f(x) = \sum_k f_k \Phi(x-x_k) */
/* Interpolation is a linear operation from vector {f_k} to vector defined on the interpolation grid */

/* Interpolation from a regular grid blob image to another regular grid image.  Blobs have the same radius. */

__global__ void BlobInterpl_Kernel(BlobType blobtype,  /* The profile function identifier */
				   LinOpType linoptype, /* Interpolation operation identifier */
				   double radius, /* radius of grid blob, after dilation */
				   double mul,	  /* multiplication factor to blob profile */
				   double dil,	  /* dilation factor to blob profile */
				   double alpha, double beta, /* Blob's possible parameter sets */
				   double vthetax, double vthetay, /* Blob grid's generating vector. -pi/2 <= theta < 0, vtheta = [cos(theta), sin(theta)]*/
				   double bgrid_splStep, /* sampling step of blob grid */
				   int bgrid_row, int bgrid_col, /* Blob virtual grid dimension, must be even */
				   const int *bgrid_mask, /* Blob grid active nodes mask */
				   int bgrid_nbNode,	  /* Number of node in blob grid */
				   double wthetax, double wthetay, /* Interpolation grid's generating vector. -pi/2 <= theta < 0, wtheta = [cos(theta), sin(theta)]*/
				   double igrid_splStep,  /* sampling step of interpolation grid */
				   int igrid_row, int igrid_col, /* Interpolation grid dimension, must be even */
				   const int *igrid_Idx, /* Blob grid active nodes index */
				   int igrid_nbNode,	  /* Number of node in interpolation grid */
				   const double *X, /* Blobs coefficients or input vector */
				   double *Y)  /* Output vector, for pixel image or for gradient image*/
{
  /* Grid maximum dimension : 65536 X 65536 = 2^32 */
  /* Thread-block maximum dimension : 512 X 512 X 64 ( <=512 ) */
  /* Launch configuration : */
  /* blockDim = (64, 1, 1), gridDim = (N, N), with N^2 = igrid_nbNode / 64 */
  /* each thread corresponds to a igrid node */

  size_t nodeIdx = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;

  /* Return immediately if current node is not active */
  if (nodeIdx >= igrid_nbNode) return;

  size_t vnodeIdx; /* Virtual (parallelogram) index of interpolation node */
  if (igrid_Idx == NULL)	/* No igrid_Idx means that all nodes are active */
    vnodeIdx = nodeIdx;
  else
    vnodeIdx = igrid_Idx[nodeIdx];
  
  /* Current node's coordinate in x and y axis, with image centered at blob node (bgrid_row/2, bgrid_col/2) */
  /* int vnodeIdxr = (int)floor(vnodeIdx * 1. / igrid_col); */
  /* int vnodeIdxc = vnodeIdx - vnodeIdxr * igrid_col; */
  /* double2 nodepos = pix2posIV(vnodeIdxr, vnodeIdxc, wthetax, wthetay, igrid_splStep, igrid_row, igrid_col); */
  double2 nodepos = nodeIdx2pos(vnodeIdx, igrid_row, igrid_col, wthetax, wthetay, igrid_splStep);

  /* SVD decomposition of matrix V = [[1, cos(theta)],[0, sin(theta)]] */
  /* V = RL * diag(l1,l2) * RR */
  /* RL=array([[cos(theta)+1,-sin(theta)],[sin(theta),1+cos(theta)]])/sqrt(2+2*cos(theta)) */
  /* RR=array([[1,1],[1,-1]])/sqrt(2) */
  /* RL<>RL^T RL*RL^T=Id */
  /* RR=RR^T, RR*RR=Id */

  /* Singular values l1 >= l2 > 0 */
  double l1 = sqrt(1+vthetax);	/* l1 = sqrt(1 + cos(theta)) */
  double l2 = sqrt(1-vthetax);	/* l2 = sqrt(1 - cos(theta)) */

  double RLt_11 = l1 / sqrt(2.); /* (1,1) element of matrix RLt */
  double RLt_12 = vthetay / l1 / sqrt(2.); /* (1,2) element of matrix RLt */
  
  /* Application of RL^T on nodepos */
  double2 Rpos = make_double2(RLt_11 * nodepos.x + RLt_12 * nodepos.y, 
			      -RLt_12 * nodepos.x + RLt_11 * nodepos.y);

  /* Contributing blobs */
  double min_1 = (Rpos.x - radius) / l1 / bgrid_splStep; /* min of eq 1 */
  double max_1 = (Rpos.x + radius) / l1 / bgrid_splStep; /* max of eq 1 */
  double min_2 = (Rpos.y - radius) / l2 / bgrid_splStep; /* min of eq 2 */
  double max_2 = (Rpos.y + radius) / l2 / bgrid_splStep; /* max of eq 2 */

  /* Be careful, convert from signed index to unsigned index */
  size_t colmin = (size_t)fmax((ceil) ((min_1 + min_2)/sqrt(2.))+bgrid_col/2, 0);
  size_t colmax = (size_t)fmin((floor)((max_1 + max_2)/sqrt(2.))+bgrid_col/2, bgrid_col-1);
  size_t rowmin = (size_t)fmax((ceil) ((min_1 - max_2)/sqrt(2.))+bgrid_row/2, 0);
  size_t rowmax = (size_t)fmin((floor)((max_1 - min_2)/sqrt(2.))+bgrid_row/2, bgrid_row-1);

  double valx = 0;
  double valy = 0;
  double valz = 0;
  double valw = 0;

  for (size_t col=colmin; col<=colmax; col++)
    for (size_t row=rowmin; row<=rowmax; row++) {      
      //double2 blobpos = pix2posIV(row, col, vthetax, vthetay, bgrid_splStep, bgrid_row, bgrid_col); /* center of this blob */
      double2 blobpos = node2pos(row, col, bgrid_row, bgrid_col, vthetax, vthetay, bgrid_splStep); /* center of this blob */
      double2 vecpos = nodepos - blobpos;
      double r = norm(vecpos); /* distance blob center to node */
      int blobIdx = bgrid_mask[row * bgrid_col + col]; /* Blob index in active set, flatten 2D array in C order */
      
      /* Things like : (r < radius && fabs(blobpos.x) < sizeObj_x /2
	 && fabs(blobpos.y) < this->sizeObj.y /2) 
	 will work, but we will need in the following blobIdx, so bgrid_mask must be
	 passed to GPU */
      
      if (r < radius && blobIdx >= 0) {
	/* procede only when inside blob radius and only for active blob */
	if (linoptype == _Img_ || linoptype == _ImgT_)	/* For interpolation operator and its adjoint */
	  switch (blobtype) {
	  case _GS_ :
	    valx += std_GS_profile(r*dil, alpha) * X[blobIdx] * mul; 
	    break;
	  case _DIFFGS_ :
	    valx += std_DIFFGS_profile(r*dil, alpha) * X[blobIdx] * mul;
	    break;
	  case _MEXHAT_ :
	    valx += std_MEXHAT_profile(r*dil, alpha) * X[blobIdx] * mul;
	    break;
	  case _D4GS_ :
	    valx += std_D4GS_profile(r*dil, alpha) * X[blobIdx] * mul;
	    break;
	  default : break;
	  }
	else if (linoptype == _Grad_) /* For gradient operator */
	  switch (blobtype) {
	  case _GS_ :		
	    valx += std_GS_dX(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    valy += std_GS_dY(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    break;
	  case _DIFFGS_ :
	    valx += std_DIFFGS_dX(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    valy += std_DIFFGS_dY(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    break;
	  case _MEXHAT_ :
	    valx += std_MEXHAT_dX(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    valy += std_MEXHAT_dY(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    break;
	  case _D4GS_ :
	    valx += std_D4GS_dX(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    valy += std_D4GS_dY(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    break;
	  default : break;
	  }	
	else if (linoptype == _GradT_)  /* For adjoint of gradient operator */
	  switch (blobtype) {
	  case _GS_ :
	    valx += -1 * std_GS_dX(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    valy += -1 * std_GS_dY(vecpos.x*dil, vecpos.y*dil, alpha) * X[bgrid_nbNode + blobIdx] * mul * dil;
	    break;
	  case _DIFFGS_ :
	    valx += -1 * std_DIFFGS_dX(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    valy += -1 * std_DIFFGS_dY(vecpos.x*dil, vecpos.y*dil, alpha) * X[bgrid_nbNode + blobIdx] * mul * dil;
	    break;
	  case _MEXHAT_ :
	    valx += -1 * std_MEXHAT_dX(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    valy += -1 * std_MEXHAT_dY(vecpos.x*dil, vecpos.y*dil, alpha) * X[bgrid_nbNode + blobIdx] * mul * dil;
	    break;
	  case _D4GS_ :
	    valx += -1 * std_D4GS_dX(vecpos.x*dil, vecpos.y*dil, alpha) * X[blobIdx] * mul * dil;
	    valy += -1 * std_D4GS_dY(vecpos.x*dil, vecpos.y*dil, alpha) * X[bgrid_nbNode + blobIdx] * mul * dil;
	    break;
	  default : break;	  
	  }
	else if (linoptype == _Derv2_)
	  switch (blobtype) {
	  case _GS_ :
	    /* valx += beta * exp(-alpha * r * r) * (pow(2*alpha*(nodepos.x - blobpos.x),2) - 2*alpha) * X[blobIdx]; */
	    /* valw += beta * exp(-alpha * r * r) * pow(2*alpha, 2) * (nodepos.x - blobpos.x)*(nodepos.y - blobpos.y) * X[blobIdx]; */
	    /* valy += beta * exp(-alpha * r * r) * (pow(2*alpha*(nodepos.y - blobpos.y),2) - 2*alpha) * X[blobIdx]; */
	    break;
	  default : break;
	  }
	else if (linoptype == _Derv2T_)
	  switch (blobtype) {
	  case _GS_ :
	    /* valx += beta * exp(-alpha * r * r) * (pow(2*alpha*(nodepos.x - blobpos.x),2) - 2*alpha) * X[blobIdx]; */
	    /* valw += beta * exp(-alpha * r * r) * pow(2*alpha, 2) * (nodepos.x - blobpos.x)*(nodepos.y - blobpos.y) * X[bgrid_nbNode + blobIdx]; */
	    /* valz += beta * exp(-alpha * r * r) * pow(2*alpha, 2) * (nodepos.x - blobpos.x)*(nodepos.y - blobpos.y) * X[bgrid_nbNode*2 + blobIdx]; */
	    /* valy += beta * exp(-alpha * r * r) * (pow(2*alpha*(nodepos.y - blobpos.y),2) - 2*alpha) * X[bgrid_nbNode*3 + blobIdx]; */
	    break;
	  default : break;
	  }
      }
    }
  
  if (linoptype == _Img_ || linoptype == _ImgT_)
    Y[nodeIdx] =  valx;
  else if (linoptype == _Grad_) {
    Y[nodeIdx] =  valx;
    Y[igrid_nbNode + nodeIdx] =  valy;
  }
  else if (linoptype == _GradT_) {
    Y[nodeIdx] = valx + valy;
  }
  else if (linoptype == _Derv2_) {
    Y[nodeIdx] =  valx;
    Y[igrid_nbNode + nodeIdx] =  valw;
    Y[igrid_nbNode * 2 + nodeIdx] =  valw;
    Y[igrid_nbNode * 3 + nodeIdx] =  valy;
  }
  else if (linoptype == _Derv2T_) {
    Y[nodeIdx] = valx + valy + valw + valz;
  }
}

#endif
