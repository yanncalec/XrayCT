#include "SpAlgo.hpp"
// L1 minimization by Iterative Soft-Thresholding

namespace SpAlgo {
  ConvMsg L0IHT_BB(const LinOp &A, const ArrayXd &Y, ArrayXd &X, 
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
    // ref : reference image
    // imsave : function handel for saving image (see CmdTools::imsave)
    // BI : BlobImage object with blob2pixel method, this is useful for saving intermediate results into a gif image
    // dimObj : desired screen image dimension
    // outpath : output path name for intermediate images

    if (verbose) {
      cout<<"-----L0 minimization by Hard Iterative Thresholding Method with BB step-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    X.setZero();
    ArrayXd gradX, gradX0, dgradX, AgradX;
    gradX.setZero(X.size());
    dgradX.setZero(X.size());
    double ngradX;

    ArrayXd dX;
    ArrayXd Z, X0 = X;
    
    ArrayXd AX, AXmY; 
    AX.setZero(Y.size());
    A.forward(X, AX);
    AXmY=AX-Y;
    
    double RdX = 1.;
    double ndX, nAdX, res, nL0, nL1;
    int niter = 0;
    double gradstep;
    //double P,C;

    while (niter < maxIter and RdX > tol) {    
      A.backward(AXmY, gradX);

      // Gradient wrt X of AL function
      gradX = A.backward(AXmY);
      ngradX = (gradX *gradX).sum();

      AgradX = A.forward(gradX);

      if (niter == 0) { // Do steepest descent at 1st iteration
	// Gradient of AL function      
	//AgradX = A.forward(gradX);
	gradstep = ngradX / (AgradX*AgradX).sum();
      }
      else {
	// Set gradstep through BB formula
	dgradX = gradX - gradX0;
	gradstep = (dX * dX).sum() / (dX * dgradX).sum();
      }
      
      // Non monotone Backtracking line-search
      int citer = 0;
      double tau = 1.;
      // Value of equivalent AL function
      double val = (AXmY-tau*gradstep*AgradX).matrix().squaredNorm()/2;
      double val0 = (AXmY).matrix().squaredNorm()/2;

      while (val > val0 - 1e-4 * tau * gradstep * ngradX and citer < 10) { // descent direction is -alpha*gradX
	tau *= 0.5;
	val = (AXmY-tau*gradstep*AgradX).matrix().squaredNorm()/2;
	citer++;
      }

      if (citer < 10)
	//BB step succeeded, update X
	Z = X - tau * gradstep * gradX;
      else {
	// BB step failed, use steepest descent
	// gradX = this->steepest_descent(X, U, alpha);
	if (verbose) 
	  cout<<"Backtrack line search failed."<<endl;

	gradstep = ngradX / (AgradX*AgradX).sum();
	Z = X - gradstep * gradX;
      }

      X = NApprx(Z, nterm);
      A.forward(X, AX);
      AXmY=AX-Y;
      dX = X - X0;
      X0 = X;
      gradX0 = gradX;
      RdX = (niter==0) ? 1. : (dX).matrix().norm() / X.matrix().norm();

      // Print convergence information
      if (verbose) {	
	res = (AX-Y).matrix().norm();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);
	printf("Iteration : %d\ttol = %1.5e\tres = %1.5e\trres = %1.5e\tL1 norm = %1.5e\tL0 norm = %1.5e\n", niter, RdX, res, res/Y.size(), nL1, nL0);
      }
      niter++;
    }

    res = (AX-Y).matrix().norm();
    nL0 = Tools::l0norm(X);

    return ConvMsg(niter, res, res, nL0);
  }
}

      // // Update P, C
      // val = AXmY.matrix().squaredNorm()/2;
      // double delta = 0.995;
      // //C = (delta * P * C + this->AL_func(X, U)) / (delta* P + 1);
      // C = (delta*P*C + val) / (delta*P + 1);
      // P = delta*P + 1;
