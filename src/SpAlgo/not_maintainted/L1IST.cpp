#include "SpAlgo.hpp"
// L1 minimization by Iterative Soft-Thresholding

namespace SpAlgo {
  ConvMsg L1IST(LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X, ArrayXi &MSupp,
		int nnz, double mu, double tol, int maxIter, int verbose)
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
      cout<<"-----L1 minimization by Soft Iterative Thresholding Method with BB gradient step-----"<<endl;
      cout<<"Solve min |Ax-b|^2 + mu*|x|_1"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"L1 fidelity penalty : "<<mu<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    BlobImage *BI = dynamic_cast<BlobImage *>(const_cast<LinOp *>(&A));

    int Freq = 10;		// Check the support changement and print message every Freq iterations.

    ArrayXd X0, dX;
    X0 = X;			
  
    ArrayXd AX = A.forward(X);
    ArrayXd gradX, gradX0, dgradX; // gradient of 1/2|Ax-y|^2
    gradX.setZero(X.size());
    gradX0.setZero(X.size());
    dgradX.setZero(X.size());

    double gradstep;		// BB gradient step

    double RdX = 1.; // Relative change in X, iteration stopping criteria
    int niter = 0;	 // Counter for iteration

    double vobj;
    double res;			   // Residual of data     
    double nL1, nL0;		   // L0 and L1 norm of X
    double nY = Y.matrix().norm(); // norm of Y
      
    ArrayXi XSupp;
    ArrayXi XSupp0 = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
    int Xdiff = X.size();

    MSupp.setZero(X.size());	// Clear support
    ArrayXi MSupp0 = XSupp0;
    int Mdiff = X.size();

    while (niter < maxIter and RdX > tol) {    
      A.backward(AX-Y, gradX);

      if (niter == 0) { // Do steepest descent at 1st iteration
	// Gradient of AL function      
	ArrayXd AgradX = A.forward(gradX);
	gradstep = (gradX*gradX).sum() / (AgradX*AgradX).sum();
      }
      else {
	// Set gradstep through BB formula
	dgradX = gradX - gradX0;
	gradstep = (dX * dX).sum() / (dX * dgradX).sum();
      }
      
      X = l1_shrink(X - gradstep * gradX, gradstep*mu, W);

      A.forward(X, AX);  
      dX = X - X0;
      X0 = X;
      gradX0 = gradX;

      RdX = (niter==0) ? 1. : (dX).matrix().norm() / X0.matrix().norm();

      if (niter % Freq == 0) {
	// Main support changement of X
	if (BI != NULL) {
	  MSupp = BI->prod_mask(X, nz);
	  Mdiff = (int)(MSupp0 - MSupp).abs().sum();
	  MSupp0 = MSupp;
	}
      }

      // Print convergence information
      if (verbose and niter % Freq == 0) {	
	res = (AX-Y).matrix().norm();
	nL1 = X.abs().sum();
	vobj = 0.5 * res * res + mu * nL1;
	nL0 = Tools::l0norm(X);
	if (BI != NULL)
	  printf("Iteration : %d\tRdX = %1.5e\tres = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\tXdiff=%d\trXdiff=%1.5e\tMdiff=%d\trMdiff=%1.5e\n", niter, RdX, res, nL1, nL0/X.size(), Xdiff, Xdiff*1./X.size(), Mdiff, Mdiff*1./X.size());
	else
	  printf("Iteration : %d\tRdX = %1.5e\tres = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\tXdiff=%d\trXdiff=%1.5e\n", niter, RdX, res, nL1, nL0/X.size(), Xdiff, Xdiff*1./X.size());
      }
      niter++;
    }

    if (niter >= maxIter and RdX > tol and verbose)
      cout<<"L1IST terminated without convergence."<<endl;      

    nL1 = X.abs().sum();
    res = (AX-Y).matrix().norm();
    vobj = 0.5 * res * res + mu * nL1;
    return ConvMsg(niter, vobj, res, nL1);
  }
}
