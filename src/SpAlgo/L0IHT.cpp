#include "SpAlgo.hpp"

namespace SpAlgo {
  ConvMsg L0IHT(LinOp &A, const ArrayXd &Y, ArrayXd &X, 
		int nterm, double tol, int maxIter, int verbose)
  {
    // A : system operator
    // Y : data in vector form
    // W : weight for L1 norm
    // X : initialization in vector form, output is rewritten on X
    // mu : L1 fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    if (verbose) {
      cout<<"-----L0 minimization by Hard Iterative Thresholding Method with BB step-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"L0 constraint : "<<nterm<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    X.setZero(A.get_dimX());

    ArrayXd gradX, gradX0, dgradX, AgradX;
    gradX.setZero(X.size());
    dgradX.setZero(X.size());

    ArrayXd dX;
    ArrayXd Z, X0 = X;
    
    ArrayXd AX; 
    AX.setZero(Y.size());
    A.forward(X, AX);
    ArrayXd AtY = A.backward(Y); 
    
    double RdX = 1.;
    double ndX, nAdX, res, nL0, nL1;
    int niter = 0;
    double gradstep;

    while (niter < maxIter and RdX > tol) {    
      A.backward(AX-Y, gradX);

      if (niter == 0) { // Do steepest descent at 1st iteration
      	// Gradient of AL function      
      	AgradX = A.forward(gradX);
      	gradstep = (gradX *gradX).sum() / (AgradX*AgradX).sum();
      }
      else {
      	// Set gradstep through BB formula
      	dgradX = gradX - gradX0;
      	gradstep = (dX * dX).sum() / (dX * dgradX).sum();
      }

      X = NApprx(X - gradstep * gradX, nterm);
      // A.Solver_CG(AtY, X, 10);
      // X = NApprx(X, nterm);
      A.forward(X, AX);
      dX = X - X0;
      X0 = X;
      gradX0 = gradX;
      RdX = (niter==0) ? 1. : (dX).matrix().norm() / X.matrix().norm();

      // TODO : Estimate the changement in support
      
      // Print convergence information
      if (verbose and niter % 25 == 0) {	
	res = (AX-Y).matrix().norm();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);
	printf("Iteration : %d\ttol = %1.5e\tres = %1.5e\trres = %1.5e\tL1 norm = %1.5e\tnon zero per. = %2.2f\n", niter, RdX, res, res/Y.size(), nL1, nL0*1./X.size());
      }
      niter++;
    }

    res = (AX-Y).matrix().norm();
    nL0 = Tools::l0norm(X);

    return ConvMsg(niter, res, res, nL0);
  }
}
