#include "SpAlgo.hpp"
// L1 minimization by Iterative Soft-Thresholding with BB step

namespace SpAlgo {
  ConvMsg MatchingPursuit(LinOp &A, const ArrayXd &Y, const ArrayXd &W, ArrayXd &X, 
			  double tol, int maxIter, int verbose)
  {
    // A : system operator
    // Y : data in vector form
    // W : footprint weight
    // X : initialization in vector form, output is rewritten on X
    // mu : L1 fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    if (verbose) {
      cout<<"-----Matching Pursuit-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    X.setZero(A.get_dimX());
    assert(W.size() == X.size());

    ArrayXd Z = Y;
    ArrayXd C = X;
    ArrayXd totoX(X.size()), totoY(Y.size());

    double Res = 1.;
    double RdX, res, nL0, nL1;
    int niter = 0;
    //std::ptrdiff_t idx;
    int idx=0; 
    double vmax=0;

    while (niter < maxIter and Res > tol) {          
      // Calculate the analysis coefficients C
      A.backward(Z, C);

      // Normalize the coefficient and find the maximum
      idx = 0; vmax = 0;
      for (int n=0; n<C.size(); n++) {
	C[n] = (W[n]>0) ? C[n]/W[n] : 0;
	if (fabs(C[n])>vmax) {
	  vmax = fabs(C[n]);
	  idx = n;
	}
      }
      // // Find the maximum element
      // C.abs().maxCoeff(&idx);
      //cout<<idx<<", "<<W[idx]<<endl;
      if (W[idx]>0) {
	// Update the synthesis coefficient
	X[idx] += C[idx] / W[idx];
	// Update the residual
	totoX.setZero(); totoX[idx] = C[idx] / W[idx];
	A.forward(totoX, totoY);
	Z -= totoY;
      }

      Res = Z.matrix().norm();
      RdX = fabs(C[idx]);

      // Print convergence information
      if (verbose and niter % 500 == 0) {	
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);
	printf("Iteration : %d\tRes=|AX-Y| = %1.5e\tRdX = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\n", niter, Res, RdX, nL1, nL0*1./X.size());
      }
      niter++;
    }
    
    //res = (A.forward(X)-Y).matrix().norm();
    nL0 = Tools::l0norm(X);
    return ConvMsg(niter, Res, Res, nL0);
  }
}
