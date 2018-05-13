#include "SpAlgo.hpp"
// L1 minimization by 

namespace SpAlgo {
  ConvMsg L1DADM(const LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X, ArrayXd &MSupp, double nz, 
		 double mu, double delta, double beta, double tau, double gamma,
		 double tol, int maxIter, int verbose)
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

    bool l2cst = delta > 0;		// true for constraint BPDN model : min |x|_1 st |Ax-b|_2 <= delta
    bool eq = (mu == 0 and delta == 0); 		// true for eq BP model 
    if (mu < 0 and delta < 0) {
      cerr<<"Wrong parameters : mu<0 and delta<0"<<endl;
      exit(0);
    }
    // if (mu == 0 and delta == 0)
    //   eq = true;
    // if (delta > 0)

    //   l2cst = delta > 0;
  
    if (verbose) {
      cout<<"-----L1 minimization by DADM-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Equality constraint : "<<eq<<endl;
      cout<<"L1 fidelity penalty : "<<mu<<endl;
      cout<<"Penalty beta : "<<beta<<endl;
      cout<<"Proximal penalty : "<<tau<<endl;
      cout<<"Lagrangian step : "<<gamma<<endl;

      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    BlobImage *BI = dynamic_cast<BlobImage *>(const_cast<LinOp *>(&A));

    ArrayXd X0, dX;
    X0 = X;			

    ArrayXd Z;			// Lagrangian multiplier
    ArrayXd R;			// Residual or the auxilary variable
    Z.setZero(Y.size()); 
    R.setZero(Y.size()); 

    ArrayXd AX = A.forward(X);

    ArrayXd grad;
    grad.setZero(X.size());
    ArrayXd toto;
    
    double RdX = 1.; // Relative change in X, iteration stopping criteria
    int niter = 0;	 // Counter for iteration

    double res;			   // Residual of data 
    double nL1;			   // L1 norm of X
    double nL0;			   // L0 norm of X
    double nY = Y.matrix().norm(); // norm of Y
    double vobj;		   // value of objective function 1/2|Ax-y|^2 + |x|_1
    // double gamma = 1.5;		   // lagrangian multiplier update step
    // double tau = 1;		   // proximal penalization, large tau for large step (fast convergence)
    double gap = 1.;		   // primal gap : |AX+R-Y|/|Y|
   
    //    while (niter < maxIter and RdX > tol and Mdiff*1./X.size() > tol) {    
    while (niter < maxIter and RdX > tol and gap > 1e-4) {    
      // Update residual
      if (!eq) {
	if (l2cst)
	  R = l2_ball_projection(Z/beta - (AX - Y), delta);
	else
	  R = (Z/beta - (AX - Y)) * beta / (1./mu + beta); 
      }

      // Update X by proximal step
      // Proximal step : <grad, x-x^k> + 1/2/tau * |x-x^k|^2
      A.backward(AX + R - Y - Z/beta, grad);
      X = l1_shrink(X - tau * grad, tau / beta, W);
      A.forward(X, AX);

      // Update lagrangian multiplier Z
      Z -= gamma * beta * (AX + R - Y);

      dX = X - X0;
      X0 = X;

      RdX = (niter==0) ? 1. : (dX).matrix().norm() / X0.matrix().norm();
      gap = (AX + R - Y).matrix().norm();
      double nZ = Z.abs().maxCoeff();

      // Print convergence information
      if (verbose) {	
	res = (AX-Y).matrix().norm();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);
	if (l2cst) {
	  printf("Iteration : %d\tRdX = %1.5e\tres = %1.5e\tdelta = %1.5e\tgap = %1.5e\tMax.Lg = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\n", niter, RdX, res, delta, gap, nZ, nL1, nL0/X.size());
	}
	else {
	  vobj = 0.5 * res * res + mu * nL1;
	  printf("Iteration : %d\tRdX = %1.5e\tvobj = %1.5e\tres = %1.5e\tgap = %1.5e\tMax.Lg = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\n", niter, RdX, vobj, res, gap, nZ, nL1, nL0/X.size());
	}
      }
      niter++;
    }

    if (RdX > tol and gap > 1e-4 and verbose)
      cout<<"L1PADM terminates without convergence."<<endl;      

    nL1 = X.abs().sum();
    res = (AX-Y).matrix().norm();
    if (l2cst)
      vobj = nL1;
    else
      vobj = 0.5 * res * res + mu * nL1;

    return ConvMsg(niter, vobj, res, nL1);
  }
}

//    int Freq = 10;		// Check the support changement and print message every Freq iterations.
    // ArrayXi XSupp;
    // ArrayXi XSupp0 = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
    // int Xdiff = X.size();

    // MSupp.setZero(X.size());	// Clear support
    // ArrayXd MSupp0 = (X.abs()>0).select(1, ArrayXd::Zero(X.size()));
    // int Mdiff = X.size();

    //   if (niter % Freq == 0) {
    // 	// Support changement of X
    // 	XSupp = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
    // 	Xdiff = (XSupp0 - XSupp).abs().sum();
    // 	XSupp0 = XSupp;
      
    // 	// Main support changement of X
    // 	if (BI != NULL) {
    // 	  MSupp = BI->prod_mask(X, nz);
    // 	  Mdiff = (int)(MSupp0 - MSupp).abs().sum();
    // 	  MSupp0 = MSupp;
    // 	}
    //   }
    //   if (verbose and niter % Freq == 0) {	
    // 	res = (AX-Y).matrix().norm();
    // 	nL1 = X.abs().sum();
    // 	nL0 = Tools::l0norm(X);
    // 	if (l2cst) {
    // 	  printf("Iteration : %d\ttol = %1.5e\tres = %1.5e\trres = %1.5e\tgap = %1.5e\tL1 norm = %1.5e\trL1 norm = %1.5e\tnon zero per. = %2.2f\tXdiff=%d\tMdiff=%d\trMdiff=%1.5e\n", niter, RdX, res, res/nY, gap, nL1, nL1/X.size(), nL0/X.size(), Xdiff, Mdiff, Mdiff*1./X.size());
    // 	}
    // 	else {
    // 	  vobj = 0.5 * res * res + mu * nL1;
    // 	  printf("Iteration : %d\ttol = %1.5e\tvobj = %1.5e\tres = %1.5e\tgap = %1.5e\trres = %1.5e\tL1 norm = %1.5e\trL1 norm = %1.5e\tnon zero per. = %2.2f\tXdiff=%d\tMdiff=%d\trMdiff=%1.5e\n", niter, RdX, vobj, res, res/nY, gap, nL1, nL1/X.size(), nL0/X.size(), Xdiff, Mdiff, Mdiff*1./X.size());
    // 	}
    //   }
    //   niter++;
    // }
