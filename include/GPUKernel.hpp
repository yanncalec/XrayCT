/* C interface for BlobInterpl_Kernel - Header file */

#ifndef _GPUKERNEL_H_
#define _GPUKERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "Types.hpp"

/*!
  GPU based blob interpolation operators and their adjoint 
*/
cudaError_t BlobInterpl_GPU(BlobType blobtype, /*!< Type of blob profile */
			    LinOpType linoptype, /*!< Type of linear operation */
			    double radius, /*!< Radius of grid blob */
			    double mul,	   /*!< multiplication factor  */
			    double dil,	   /*!< dilation factor  */
			    double alpha, double beta, /*!< Blob's parameters */
			    double vthetax, double vthetay, /*!< Grid's generating vector, the corresponding angle : -pi/2 <= theta < 0*/
			    double bgrid_splStep, /*!< sampling step of blob grid */
			    int bgrid_row, int bgrid_col, /*!< Blob grid dimension, even X even */		   
			    const int *bgrid_mask,		/*!< mask for active nodes of blob grid, GPU memory  */
			    int bgrid_nbNode,		/*!< total number of active nodes in blob grid  */
			    double wthetax, /*!< Gradient grid's generating vector. \f$ \cos(\theta), -\pi/2 \leq \theta < 0 \f$ */
			    double wthetay, /*!< Gradient grid's generating vector. \f$ \sin(\theta), -\pi/2 \leq \theta < 0 \f$ */
			    double igrid_splStep,  /*!< sampling step of gradient grid */
			    int igrid_row, int igrid_col, /*!< Gradient grid dimension, even X even */		   
			    const int *igrid_Idx,		/*!< Index for interpolation grid active nodes, GPU memory  */
			    int igrid_nbNode,		/*!< total number of active nodes in interploation grid */
			    const double *d_X, /*!< Blobs coefficients, GPU memory */
			    double *d_Y); /*!< Output interpolated image, GPU memory */	

/*!
  GPU based blob-driven projector and their adjoint, the meaning of uncommented variables are either trivial or can be found somewhere else in the code.
*/
cudaError_t BlobProjector_GPU(BlobType blobtype,  /* The profile function identifier */
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
			      const double *d_X,
			      const bool *d_MX,
			      double *d_Y,
			      const bool *d_MY,
			      bool forward);

/*!
  GPU based pixel-driven projector and their adjoint 
*/
cudaError_t PixDrvProjector_GPU(bool fanbeam, 
				int nbProj, 
				const int *d_projset, 
				const double *d_pSrc, 
				const double *d_rSrc, 
				const double *d_rDet,   
				double sizeDet, int pixDet, 
				double spObj,
				int row, int col, /* Blob grid dimension, even X even */   
				const double *d_X,
				double *d_Y,
				bool forward);

/*!
  GPU based forward projector norm estimation
 */
cudaError_t BlobProjector_NormEstimator_GPU(BlobType blobtype,
					    double radius, 
					    double mul, 
					    double dil,
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
					    double *d_Y);

// /*!
//   GPU based matching pursuit projector, only backward projection
//  */
// cudaError_t BlobMatching_GPU(BlobType blobtype,  /* The profile function identifier */
// 			     double radius, 
// 			     double mul, 
// 			     double dil, /* radius is that of dilated blob */
// 			     double alpha, double beta,			      
// 			     bool fanbeam, 
// 			     int nbProj, 
// 			     const int *d_projset, 
// 			     const double *d_pSrc, 
// 			     const double *d_rSrc, 
// 			     const double *d_rDet,   
// 			     double sizeDet, int pixDet, 
// 			     const double *d_T, size_t dimT,
// 			     double thetax, double thetay, /* Blob grid's generating vector, the corresponding angle : -pi/2 <= theta < 0*/
// 			     double splStep, /* sampling step of blob grid */
// 			     int row, int col, /* Blob grid dimension, even X even */   
// 			     const int *d_Idx,
// 			     int nbNode,
// 			     const double *d_X,
// 			     const bool *d_M,
// 			     const double *d_Y,
// 			     double *d_Z);

// /*!
//   GPU based MART operator, obsolete
// */
// cudaError_t MARTOp_GPU(bool fanbeam, 
// 		       int nbProj, 
// 		       const int *d_projset, 
// 		       const double *d_pSrc, 
// 		       const double *d_rSrc, 
// 		       const double *d_rDet,   
// 		       double sizeDet, int pixDet, 
// 		       double spObj,
// 		       int row, int col, /* Blob grid dimension, even X even */   
// 		       const double *d_X,
// 		       double *d_Y);

#endif
