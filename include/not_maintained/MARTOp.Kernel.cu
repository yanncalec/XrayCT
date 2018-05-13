/*  -*- C -*-  */
/* On-the-fly projector based on pixel representation */
/* GPU implementation */

#ifndef _MARTOP_KERNEL_H_
#define _MARTOP_KERNEL_H_

#include "GPUDevFunc.cu"
#include "GPUTypes.hpp"

__shared__ double val[MAXNBPROJ]; /* Temporary variable to save projection or backprojection value */  

__global__ void MARTOp_Kernel(bool fanbeam, 
			      const int *projset,
			      const double *pSrc, 
			      const double *rSrc, 
			      const double *rDet, 
			      double sizeDet, int pixDet, 
			      double spObj, /* sampling step of cartesian grid */
			      int row, int col, /* Cartesian grid dimension, even X even */   
			      const double *X, 
			      double *Y)
{
  size_t pixIdx = blockIdx.x * gridDim.y + blockIdx.y; /* The pixel represented by current kernel */
  size_t projIdx = projset[threadIdx.x]; /* Projection set represented by current kernel, maximum 512 projections can be treated simultaneously */
  size_t nbProj = blockDim.x; /* block dimension equals to the number of projections */

  double spDet = sizeDet / pixDet; /* Size of detector pixel */

  extern __shared__ double val[]; /* Temporary variable to save the backprojection value */ 

  if (pixIdx >= row * col)	/* Return immediately if current kernel is out of range */
    return;

  /* backward : initialize shared memory */
  size_t tt = (size_t)(ceil(MAXNBPROJ * 1. / nbProj)); 
  for (size_t j=0; j<tt; j++)
    if (threadIdx.x * tt+j < MAXNBPROJ)
      val[threadIdx.x * tt+j] = 0;

  double2 S;			/* Source position */
  double2 D;			/* Center of detector plan */
  double2 P;			/* Pix node position */
  double2 A;	     /* Arriving point's position */
  double2 L; /* Detector left-border position, from source's viewpoint */

  D.x = rDet[projIdx] * cos(pSrc[projIdx] + M_PI); /* Center of detector plan is always at opposite side of source */
  D.y = rDet[projIdx] * sin(pSrc[projIdx] + M_PI);

  double2 dU;	 /* Unitary vector of the detector plan, starting from
		    left border and going to right border. The left and
		    right is from source's viewpoint */
  dU.x = -sin(pSrc[projIdx]); dU.y = cos(pSrc[projIdx]); /* Without detector plan's
							    rotation, dU is the rotation
							    by pi/2 in anti-clockwise
							    sens of vector OS */

  L = D - sizeDet/2 * dU; /* By definition, L = D - sizeDet/2 * dU */

  /* The coordinate of current pixel */
  P = pix2pos(pixIdx, row, col, spObj); 

  if(fanbeam){				      /* Fan beam */
    S.x = cos(pSrc[projIdx]) * rSrc[projIdx]; /* Source position in fanbeam */
    S.y = sin(pSrc[projIdx]) * rSrc[projIdx];
  }

  double coeff = -1;			 /* contribution indicator */
  int dpixL, dpixR;			 /* detector pixel index influenced by an image pixel */
  double Al, Ar;			 /* image pixel arriving position on detector, left and right from source pov */
  if (fanbeam) {
    double2 rg = cercle_projection(S, P, spObj*sqrt(2.)/2, L, dU); /* treat the image pixel as a cercle */
    Al = rg.x; Ar = rg.y;
  }
  else {
    Al = dot(P-L, dU) - spObj*sqrt(2.)/2;
    Ar = dot(P-L, dU) + spObj*sqrt(2.)/2;
  }
  
  if (Al<sizeDet && Ar>0){ /* Ray must be on detector area */
    /* Conversion of Al and Ar to detector pixels */
    int dpixL = (int)floor(Al / spDet); /* Detector pixel including Al */
    int dpixR = (int)ceil(Ar / spDet); /* Detector pixel including Ar */
    dpixL = (dpixL<0) ? 0 : dpixL;
    dpixR = (dpixR>pixDet) ? pixDet : dpixR;

    for (int dpix = dpixL; dpix < dpixR; dpix++) {
      if(!fanbeam) {		/* source position when parallel beam */
	S.x = cos(pSrc[projIdx])*rSrc[projIdx] + dU.x * ((dpix+0.5) * spDet - sizeDet/2);
	S.y = sin(pSrc[projIdx])*rSrc[projIdx] + dU.y * ((dpix+0.5) * spDet - sizeDet/2);
      }

      A = L + (dpix+0.5)*spDet*dU; /* center of current detector pixel bin */
      if (fabs(S.x - A.x)<1e-8 || fabs(S.y - A.y)<1e-8) { /* SA line is parallel to X or Y axis, special treatement */
	//if (S.x == A.x || S.y == A.y) {
	coeff = fabs(dot(P-L, dU) - (dpix+0.5)*spDet) < spObj/2 ? 1 : -1;
      }
      else {
	double amin, amax, bmin, bmax, a1, a2, b1, b2, la, lb;
	double SA = norm(S-A);
	a1=(P.x + spObj/2 - S.x) / (A.x-S.x);
	a2=(P.x - spObj/2 - S.x) / (A.x-S.x);
	b1=(P.y + spObj/2 - S.y) / (A.y-S.y);
	b2=(P.y - spObj/2 - S.y) / (A.y-S.y);
	amin = fmin(a1,a2); amax = fmax(a1,a2);
	bmin = fmin(b1,b2); bmax = fmax(b1,b2);
	la = fmax(amin, bmin); 
	lb = fmin(amax, bmax);
	coeff = (la < lb) ? 1 : -1;
      }
      if (coeff > 0) {		/* current detector pixel contributes to current image pixel */
	if (isinf(X[dpix + pixDet * threadIdx.x])) {
	  Y[pixIdx] = 0;
	  return;
	}
	else
	  val[threadIdx.x] += X[dpix + pixDet * threadIdx.x];
      }
    }
  }

  /* Reduction  */
  __syncthreads();
    
  for (size_t s=1; s<nbProj; s*=2) {
    size_t idx = 2 * s * threadIdx.x;
    if (idx < nbProj)
      val[idx] += val[idx+s];
    __syncthreads();
  }
    
  if (threadIdx.x == 0) {
    if (isinf(val[threadIdx.x]))
      Y[pixIdx] = 0;
    else
      Y[pixIdx] *= exp(val[threadIdx.x]);
    //Y[pixIdx] = isfinite(Y[pixIdx]) ? Y[pixIdx] : 0;
  }
}


#endif

