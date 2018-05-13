// This file contains some important constants for GPU calculation

#ifndef _GPUTYPES_H_
#define _GPUTYPES_H_

#define BLOBINTERPL_BLOCKSIZE 64 	/* Block dimension for BlobInterpl */
#define MAXNBPROJ 512		/* Maximum number of projections (<= 1024) */
/* On architecture >= 2.0, one can calculate up to 1024 projections (max nb. of threads in a block) */

#define MAXDPIX 128		/* Maximum number of detector bin influenced by a single blob */
/* Small value could raise Lauch Failure error if the blob radius becomes large, but big value can consume more resources. */

#define N_DIFFGS 10	/* The number of approximation (asymptotic expansion) terms in Diff-Gaussian blob */
/* Value > 10 is almost prohibitive for speed reason */

/*!
  - _GS_ : Gaussian
  - _DIFFGS_ : Diff-Gaussian
  - _MEXHAT_ : Mexican Hat (negative 2nd order derivative of Gaussian)
  - _D4GS_ : Fourth order derivative of Gaussian
  - _KB_ : Kaisser Bessel (not implemented)
  - _CBS_ : cublic B-Spline (not implemented) 
*/
typedef enum{_GS_, _DIFFGS_, _MEXHAT_, _D4GS_, _KB_, _CBS_} BlobType;

/*!
  - _Proj_, _ProjT_ : projector
  - _Img_, _ImgT_ : simple interpolation of f(x)
  - _Grad_, _GradT_ : gradient interpolation and its adjoint (d/dX, d/dY)f(x), with d/dX derivative in X direction
  - _Derv2_, _Derv2T_ : second order derivative interpolation and its adjoint (d^2/dX^2, 2 * d^2/dXdY, d^2/dY^2)f(x) 
*/
typedef enum{_Proj_, _ProjT_, _Img_, _ImgT_, _Grad_, _GradT_, _Derv2_, _Derv2T_, _UnknownOp_} LinOpType;

#endif
