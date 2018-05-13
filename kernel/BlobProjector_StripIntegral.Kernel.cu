/*  -*- C -*-  */
/* On-the-fly blob based projector */
/* GPU implementation */

#ifndef _BLOBPROJECTOR_STRIPINTEGRAL_KERNEL_H_
#define _BLOBPROJECTOR_STRIPINTEGRAL_KERNEL_H_

#include "GPUDevFunc.cu"
#include "BlobFunc.cu"
#include "GPUTypes.hpp"

extern __shared__ double val[]; /* Temporary variable to save projection or backprojection value */  

__global__ void BlobProjector_StripIntegral_Kernel(BlobType blobtype,  /* The profile function identifier */
						   double radius, 
						   double mul, 
						   double dil, /* radius is that of dilated blob */
						   double alpha, double beta,
						   bool fanbeam, 
						   const int *projset,
						   const double *_pSrc, 
						   const double *_rSrc, 
						   //const double *rtDet,
						   const double *_rDet, 
						   double sizeDet, int pixDet,
						   const double *T, size_t dimT, /* The table T is standard : dilation and multiplication factors are not included */				     
						   double thetax, double thetay, /* Blob grid's generating vector, the corresponding angle : -pi/2 <= theta < 0*/
						   double splStep, /* sampling step of blob grid */
						   int row, int col, /* Blob grid dimension, even X even */   
						   const int *Idx, /* row index of active nodes in the parallelogram */
						   int nbNode,
						   const double *X, 
						   const bool *MX,
						   double *Y, 
						   const bool *MY,
						   bool forward)
{
  size_t blobIdx = blockIdx.x * gridDim.y + blockIdx.y; /* The blob node represented by current kernel */
  size_t projIdx = projset[threadIdx.x]; /* Projection set represented by current kernel, maximum 512 projections can be treated simultaneously */
  size_t nbProj = blockDim.x; /* block dimension equals to the number of projections */

  extern __shared__ double val[]; /* Temporary variable to save the backprojection value */ 

  if (blobIdx >= nbNode)
    return;

  if (forward) {	   /* forward : return immediately if blob coeff is 0 or not in the mask*/
    if (fabs(X[blobIdx]) == 0.0 || !MX[blobIdx])
      return;
  }
  else {			/* backward */
    if (!MY[blobIdx])		/* return immediately if blob is not in the mask */
      return;
    else {			/* otherwise initialize the shared memory */
      size_t tt = (size_t)(ceil(MAXNBPROJ * 1. / nbProj)); 
      for (size_t j=0; j<tt; j++)
	if (threadIdx.x * tt+j < MAXNBPROJ)
	  val[threadIdx.x * tt+j] = 0;
    }
  }

  double Al, Ar;		/* Influenced interval on detector */
  double Tval[MAXDPIX];		/* The maximum of detector pixels influenced by current blob */
  double spDet = sizeDet / pixDet; /* Size of detector pixel */

  double pSrc = _pSrc[projIdx];
  double rSrc = _rSrc[projIdx];
  double rDet = _rDet[projIdx];

  double2 S;			/* Source position */
  double2 D;			/* Center of detector plan */
  double2 P;			/* Blob node position */
  double2 L; /* Detector left-border position, from source's viewpoint */
  double A0; /* distance from L to arriving point of P on detector */
  double2 SP, SL;		/* Vectors from S to P and from S to L */

  D.x = rDet * cos(pSrc + M_PI); /* Center of detector plan is always at opposite side of source */
  D.y = rDet * sin(pSrc + M_PI);

  double2 dU;	 /* Unitary vector of the detector plan, starting from
		    left border and going to right border. The left and
		    right is from source's viewpoint */
  dU.x = -sin(pSrc); dU.y = cos(pSrc); /* Without detector plan's
					  rotation, dU is the rotation
					  by pi/2 in anti-clockwise
					  sens of vector OS */
  //dU = rotation(dU, rtDet);	/* dU of inclined detector plan */ 

  L = D - sizeDet/2 * dU; /* By definition, L = D - sizeDet/2 * dU */

  /* The coordinate of current node(blob) */
  /* int rr = (int)floor(Idx[blobIdx] * 1. / vshape.x); */
  /* int cc = Idx[blobIdx] - rr * vshape.x; */
  /* P.x = splStep * ((cc-vshape.x/2) + (rr-vshape.y/2) * vtheta.x); */
  /* P.y = splStep * (rr-vshape.y/2) * vtheta.y; */
  P = nodeIdx2pos(Idx[blobIdx], row, col, thetax, thetay, splStep);
      
  if(fanbeam){				      /* Fan beam */
    S.x = cos(pSrc) * rSrc; /* Source position in fanbeam */
    S.y = sin(pSrc) * rSrc;
    SP = P - S;			/* Vector from S to P */
    SL = L - S;

    double sin_theta = radius / norm(SP); /* theta is the angle formed by the tangent vectors and SP, it's included in [0, pi/2] */
    double cos_theta = sqrt(1 - sin_theta*sin_theta);
	
    double2 SP1, SP2;	/* Tangent vectors with blob P */
    SP1.x = cos_theta * SP.x - sin_theta * SP.y; /* Rotation of SP by theta */
    SP1.y = sin_theta * SP.x + cos_theta * SP.y;
    SP2.x = cos_theta * SP.x + sin_theta * SP.y; /* Rotation of SP by -theta */
    SP2.y = -sin_theta * SP.x + cos_theta * SP.y;

    /* Arriving positions of SP1 SP2 on detector plan */
    double l1 = -cross(SP1, SL)/cross(SP1,dU); /* l1 is the distance on detector plan from L to the arriving point of SP1 */
    double l2 = -cross(SP2, SL)/cross(SP2,dU);
    Al = fmin(l1,l2); 
    Ar = fmax(l1,l2); 
  }
  else { /* Parallel beam */
    A0 = dot(P-L, dU);
    Al = A0 - radius;		 
    Ar = A0 + radius;
  }

  if (Al<sizeDet && Ar>0){ /* Ray must be on detector area */
    /* Conversion of Al and Ar to detector pixels */
    int dpixL = (int)floor(Al / spDet); /* Detector pixel including Al */
    int dpixR = (int)ceil(Ar / spDet); /* Detector pixel including Ar */
    dpixL = (dpixL<0) ? 0 : dpixL;
    dpixR = (dpixR>pixDet) ? pixDet : dpixR;

    double vv;			/* blob center to ray distance */
    for (int dpix = dpixL; dpix <= dpixR; dpix++) {
      if (fanbeam) {
	double2 SE = SL + dpix * spDet * dU; /* Vector from S to E.  E is the detector pixel dpix's left border. SE = SL + LE */
	double2 PP = (dot(SP, SE)/dot(SE,SE)) * SE - SP; /* Vector from P to SE, orthogonal to SE */
	vv = (dot(PP,dU)<=0) ? -norm(PP) : norm(PP); /* Signed projection distance from P to SE :
							negative at left side of A0, and positive at right side.
							this conforms to the SL view point */
      }
      else
	vv = dpix*spDet-A0;
      /* 
	 Important relation : Let \phi(x) = b*\phi0(a*x), then
	 1. Abel \phi(t) = b / a * Abel \phi0(a*t)
	 2. Strip integral : \int_{-\infty}^B Abel \phi(t) dt = b/(a^2) \int_{-\infty}^{a*B} Abel \phi0(t) dt
      */
      /* 
	 The table T is made from the standard blob (ie, the dilation and multiplication factors are not included).
	 We shall look up the standard table T with value dil*vv, therefore pass the argument dil*vv/R0, R0 is the stand blob's radius. 
	 Remark that dil*vv/R0 = vv/radius.
      */

      Tval[dpix-dpixL] = lookuptable(T, dimT, vv/radius) * mul / (dil*dil);

      if (dpix>dpixL) {
	if (forward){ 	/* forward operation */
	  if (MY[dpix-1 + pixDet * threadIdx.x])	/* dpix are borders of pixels */
	    atomicAddDouble(&Y[dpix-1 + pixDet * threadIdx.x],  (Tval[dpix-dpixL]-Tval[dpix-dpixL-1]) * X[blobIdx]);
	}
	else			/* backward */
	  val[threadIdx.x] += (Tval[dpix-dpixL]-Tval[dpix-dpixL-1]) * X[dpix-1 + pixDet * threadIdx.x];
      }
    }

    if (!forward) {
      /* Reduction  */
      __syncthreads();
    
      for (size_t s=1; s<nbProj; s*=2) {
	size_t idx = 2 * s * threadIdx.x;
	if (idx < nbProj)
	  val[idx] += val[idx+s];
	__syncthreads();
      }
    
      if (threadIdx.x == 0)
	Y[blobIdx] = val[threadIdx.x];
    }
  }
}

#endif

/* Other possible reduction algorithm for backward projection  */

/* for (unsigned int s=1; s<BLOCKSIZE; s*=2){ */
/*   if (projIdx % (2*s) == 0) */
/*     val[projIdx] += val[projIdx+s]; */
/*   __syncthreads(); */
/* } */

/* for (unsigned int s=BLOCKSIZE/2; s>0; s>>=1) { */
/*   if (projIdx < s) { */
/*     val[projIdx] += val[projIdx + s]; */
/*   } */
/*   __syncthreads(); */
/* } */

/* for (unsigned int s=BLOCKSIZE/2; s>32; s>>=1) */
/*   { */
/*     if (projIdx < s) */
/*       val[projIdx] += val[projIdx + s]; */
/*     __syncthreads(); */
/*   } */
/* if (projIdx < 32) */
/*   { */
/*     val[projIdx] += val[projIdx + 32]; */
/*     val[projIdx] += val[projIdx + 16]; */
/*     val[projIdx] += val[projIdx + 8]; */
/*     val[projIdx] += val[projIdx + 4]; */
/*     val[projIdx] += val[projIdx + 2]; */
/*     val[projIdx] += val[projIdx + 1]; */
/*   } */



/* if (T != NULL) */
/* 	Tval[dpix-dpixL] = lookuptable(T, dimT, vv/radius) * (stripint ? 1/dil : 1) * mul / dil; */
/* else { */
/* 	switch (blobtype) { */
/* 	case _GS_ : */
/* 	  Tval[dpix-dpixL] = std_GS_Abel(vv*dil, alpha) * mul / dil; */
/* 	  break; */
/* 	case _DIFFGS_ : */
/* 	  Tval[dpix-dpixL] = std_DIFFGS_Abel(vv*dil, alpha) * mul / dil; */
/* 	  break; */
/* 	case _MEXHAT_ : */
/* 	  Tval[dpix-dpixL] = std_MEXHAT_Abel(vv*dil, alpha) * mul / dil; */
/* 	  break; */
/* 	case _D4GS_ : */
/* 	  Tval[dpix-dpixL] = std_D4GS_Abel(vv*dil, alpha) * mul / dil; */
/* 	  break; */
/* 	default : break; */
/* 	} */
/* } */
