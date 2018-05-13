#ifndef _SPALGO_H_
#define _SPALGO_H_

#include "Algo.hpp"
#include "AcqConfig.hpp"
#include "BlobImage.hpp"
#include "LinOp.hpp"
#include "SpLinOp.hpp"
#include "LinSolver.hpp"
#include "Tools.hpp"
//#include "BlobProjector.hpp"

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
//#include <cuda_runtime.h>

using namespace Eigen;
using namespace std;

//! Collection of sparsity related functions and sparsity based reconstruction algorithms. 
namespace SpAlgo{
  // Sparsity related tools
  ArrayXd GTVNorm(const ArrayXd &, int); //!< Generalized total variation norm
  ArrayXd TV_normalize(ArrayXd &, int);

  ArrayXd l1_shrink(const ArrayXd &, double); //!< L1 shrinkage operator
  ArrayXd l1_shrink(const ArrayXd &, double, const ArrayXd &); //!< L1 reweighted shrinkage operator
  ArrayXd l2_shrike(const ArrayXd &V, double mu, int gradRank); //!< L2 shrinkage operator, gradRank is the space dimension (2 or 3)
  ArrayXd l2_shrike(const ArrayXd &V, double mu, int gradRank, const ArrayXd &W); //!< L2 reweighted shrinkage operator, gradRank is the space dimension (2 or 3)

  ArrayXd l2_ball_projection(const ArrayXd &X, double radius); //!< Projection onto l2 ball
  ArrayXd linf_ball_projection(const ArrayXd &X, double radius); //!< Projection onto l-inf ball

  ArrayXd NApprx(const ArrayXd &X, int nterm); //!< Best n-term approximation
  ArrayXi NApprx_support(const ArrayXd &X, int nterm);  //!< The support (0 or 1) of the best n-term approximation
  ArrayXd NApprx_support_double(const ArrayXd &X, int nterm); //!< The support (0 or 1) of the best n-term approximation in double format

  // ArrayXd TVe_X(const ArrayXd &, int, double); // TVGradient
  // ArrayXd TVe_X_normalize(ArrayXd &, int, double); // TVGradient

  // Sparsity based algorithms
  // TV family
  ConvMsg TVADM(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X, 
		double delta, bool aniso,
		double mu, double beta,
		double tol_cg, int maxIter_cg,
		double tol, int maxIter, int verbose);

  ConvMsg TVAL3(LinOp &A, LinOp &G, const ArrayXd &W, const ArrayXd &Y,	ArrayXd &X,
		//		ArrayXd &Nu, ArrayXd &Lambda,
		double delta, bool aniso, bool nonneg, double mu, double beta,
		double tol_Inn, double tol_Out, double tol_gap, int maxIter, int verbose);
  // ConvMsg TVAL3(LinOp &A, LinOp &G, const ArrayXd &W, const ArrayXd &Y,
  // 		ArrayXd &X, ArrayXd &Nu, ArrayXd &Lambda,
  // 		bool eq, bool aniso, bool nonneg, double mu, double beta,
  // 		double tol_Inn, double tol_Out, int maxIter, int verbose);

  // L1 family
  ConvMsg L1FISTA_Homotopy(LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X,
			   double mu, double tol, int maxIter,
			   double stol, double sparsity, 
			   int hIter, double incr, int debias, bool noapprx, int verbose, int Freq);

  ConvMsg L1FISTA(LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X,
		  double mu, double tol, int maxIter, 
		  double stol, double sparsity, int verbose, int Freq,
		  void (*savearray)(const ArrayXd &, const string &) = NULL, const string * = NULL);

  ConvMsg L1PADM(LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X, ArrayXd &Z, ArrayXd &R,
		 //  ConvMsg L1PADM(LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X,
		 double delta, double mu, double beta, double tau, double gamma,
		 double tol, int maxIter, double stol, double sparsity, int verbose, int Freq);

  // TV-L1 family
  ConvMsg TVL1ADM(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X, ArrayXd &U, ArrayXd &V,
		  double delta, double gamma, double muA, double muG, double muI, bool aniso,
		  double tol_cg, int maxIter_cg, double tol, int maxIter, int verbose);

  ConvMsg TVL1IADM(LinOp &A, LinOp &G, const ArrayXd &W, const ArrayXd &Y,
		   ArrayXd &X, ArrayXd &Nu, ArrayXd &Lambda, ArrayXd &Omega,
		   bool eq, bool aniso, double mu, double gamma, double alpha, double beta,
		   double tol_Inn, double tol_Out, int maxIter, int verbose);

  ConvMsg TVL1Prxm(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X,
		   double delta, double gamma, 
		   double muA, double muG, double muI, bool aniso,
		   double tol, int maxIter, 
		   double stol, double sparsity, int verbose, int Freq,
		   void (*savearray)(const ArrayXd &, const string &) = NULL, const string * = NULL);

  ConvMsg TVL1SPL(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X, 
		  bool aniso, double alpha, double beta, 
		  double tau1, double tau2,
		  double tol, int maxIter,  
		  double stol, double sparsity, int verbose);

  ConvMsg TVL1Prxm_Homotopy(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X, double gamma, 
			    double muA, double muG, double muI, bool aniso, 
			    double tol, int maxIter, double sparsity, 
			    double stol, int hIter, double incr, int debias, bool noapprx, 
			    int verbose, int Freq);

  // Greedy methods
  ConvMsg MatchingPursuit(LinOp &A, const ArrayXd &Y, const ArrayXd &W, ArrayXd &X, 
			  double tol, int maxIter, int verbose);

  ConvMsg CoSaMP(LinOp &A, const ArrayXd &Y, ArrayXd &X,  double sparsity, 
		 double tol, int maxIter, double cgtol, int cgmaxIter, int verbose);

  ConvMsg L0IHT(LinOp &A, const ArrayXd &Y, ArrayXd &X, 
		int nterm, double tol, int maxIter, int verbose);

};


#endif

  // ConvMsg L1AL3(LinOp &A, const ArrayXd &W, const ArrayXd &Y,
  // 		ArrayXd &X, ArrayXd &Nu, ArrayXd &Lambda,
  // 		bool eq, bool nonneg, double mu, double beta,
  // 		double tol_Inn, double tol_Out, int maxIter, int verbose);

  // ConvMsg L1IST(LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X, ArrayXi &MSupp,
  // 		double nz, double mu, double tol, int maxIter, int verbose);
