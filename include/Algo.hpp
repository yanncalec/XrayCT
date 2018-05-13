#ifndef _ALGO_H_
#define _ALGO_H_

// Collection of classic pixel based reconstruction algorithms

#include "AcqConfig.hpp"
#include "LinOp.hpp"
#include "Projector.hpp"
#include "Tools.hpp"

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
//#include <fftw3.h>
//#include <cuda_runtime.h>

using namespace Eigen;
using namespace std;

//! Edge-preserving prior (EPP) type indicator
typedef enum {_EPP_GM_, _EPP_HS_, _EPP_GS_, _EPP_GR_} EPPType;

//! Collection of classical reconstruction algorithms. 
namespace Algo{

  //! Landwebber iteration (not implemented).
  /*!
    Solve \f$ \min \|Ax - b\| \f$ (with constraint \f$ x\geq 0\f$).
    using projected gradient descent method.
  */
  ConvMsg Landwebber(LinOp &P, //!< Projector object
		     const ArrayXd &B, //!< Sinogram vector
		     ArrayXd &X, //!< Initialization, also the output solution
		     bool nonneg,	    //!< Positive constraint 
		     double tol, 
		     int maxIter, 
		     bool verbose);

  // //! Maximization of entropy (not implemented)
  // /*!
  //   Solve \f$ \min \sum x_i\log(x_i) s.t. Ax=b\f$
  //   by MART (Multiplication ART) method.
  //   The solution is unique and positive.
  //  */
  
  // ConvMsg MART(LinOp &A, 	//!< Projector
  // 	       MARTOp &M, 	//!< MART backprojector
  // 	       const ArrayXd &Y,
  // 	       ArrayXd &X, 	//!< Initialization, must be strictly positive
  // 	       double tau,	//!< MART exponential factor
  // 	       double tol, int maxIter, int verbose);
      
  double ARTUR_delta(double t, EPPType priortype, double alpha);
  double ARTUR_phi(double t, EPPType priortype, double alpha);

  //! ARTUR (or Half-Quadratic) algorithm for Edge-Preserving reconstruction.
  /*!
    Solve \f$ \min \|Ax - b\|^2 + \mu\sum_k \phi(D_n x) \f$
    using the ARTUR algorithm.
    Ref: "Deterministic edge-preserving regularization in computed imaging"
   */
  ConvMsg ARTUR(LinOp &A, 	//!< Projector object
		LinOp &G, 	//!< Gradient operator object
		const ArrayXd &Y, //!< Sinogram vector
		ArrayXd &X,	  //!< Initialization, also the output solution
		EPPType priortype, //!< Edge-Preserving Prior type indicator
		double alpha,	   //!< Parameter of EPP function $\phi(\cdot; \alpha)$
		double mu,	   //!< EPP weight
		double cgtol,	   //!< CG iteration precision
		int cgmaxIter,	   //!< CG maximum number of iterations
		double tol,	   //!< Global precision
		int maxIter,	   //!< Global maximum number of iterations 
		int verbose	   //!< Print convergence message
		);

  double BesovPrior(const ArrayXd &, double, double=0);
  ArrayXd BesovPriorGrad(const ArrayXd &X, double, double=0);

  //! Reconstruction by Besov norm regularization using Wavelet transform
  /*!
    Solve \f$ min \|AWx - b\|^2 + \mu\|x\|_B \f$
    using the nonlinear CG algorithm.
    \f$ AW \f$ is the system operator composed by projector A and Wavelet synthesis operator W.
    \f$\|x\|_B\f$ is the Besov norm on the wavelet coefficient \f$x\f$.
    Ref: "Bayesian multiresolution method for local tomography in dental x-ray imaging"
   */
  ConvMsg WaveletBesov(LinOp &AW, //!< System operator
		       const ArrayXd &Y, //!< Sinogram vector
		       ArrayXd &X,	  //!< Initialization, also the output solution (Wavelet coeffcient)
		       double mu,	  //!< Besov norm weight
		       double besov_p,	  //!< Besov norm order parameter (must be bigger than 1, close to 1 is better)
		       double besov_s,	  //!< Besov norm scale parameter (0 is best choice)
		       double tol,	   //!< Global precision
		       int maxIter,	   //!< Global maximum number of iterations 
		       int verbose	   //!< Print convergence message
		       );

};

#endif
