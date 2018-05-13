#include "SpAlgo.hpp"
// L1 minimization by 

namespace SpAlgo {
  ConvMsg L1DADM(const LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X, ArrayXd &MSupp, double nz, 
		 double mu, double beta, double gamma, double tol, int maxIter, int verbose)
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

    bool eq = (mu == 0); 		// true for eq BP model 
  
    if (verbose) {
      cout<<"-----L1 minimization by DADM-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Equality constraint : "<<eq<<endl;
      cout<<"L1 fidelity penalty : "<<mu<<endl;
      cout<<"Penalty beta : "<<beta<<endl;
      cout<<"Lagrangian step : "<<gamma<<endl;

      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    BlobImage *BI = dynamic_cast<BlobImage *>(const_cast<LinOp *>(&A));

    //X.setZero();
    ArrayXd X0, dX;
    X0 = X;			

    ArrayXd R, Z;
    Z.setZero(X.size()); 
    R.setZero(Y.size()); 

    ArrayXd AX = ArrayXd::Zero(Y.size()); //A.forward(X);
    ArrayXd AtR = ArrayXd::Zero(X.size());

    ArrayXd grad, Atgrad;
    grad.setZero(Y.size());
    Atgrad.setZero(X.size());

    ArrayXd totoX, totoY;
    totoX.setZero(X.size());
    totoY.setZero(Y.size());
    
    double RdX = 1.; // Relative change in X, iteration stopping criteria
    int niter = 0;	 // Counter for iteration

    double res;			   // Residual of data 
    double nL1;			   // L1 norm of X
    double nL0;			   // L0 norm of X
    double nY = Y.matrix().norm(); // norm of Y
    double vobj;		   // value of objective function 1/2|Ax-y|^2 + |x|_1
    // double gamma = 1.5;		   // lagrangian multiplier update step
    // double tau = 1;		   // proximal penalization, large tau for large step (fast convergence)
    double gap = 1., primal_gap, dual_gap, delta_gap;		   // gaps
   
    //    while (niter < maxIter and RdX > tol and Mdiff*1./X.size() > tol) {    
    while (niter < maxIter and RdX > tol) {    
      // Update Z
      totoX = AtR + X/beta;
      Z = linf_ball_projection(totoX, 1);

      // Update R
      A.forward(AtR + X/beta - Z, totoY);
      grad = mu * R - Y + beta * totoY;
      double ngrad = (grad*grad).sum();
      A.backward(grad, Atgrad);
      double gradstep = ngrad / (mu * ngrad + beta * (Atgrad*Atgrad).sum());
      R -= gradstep * grad;

      A.backward(R, AtR);

      // Update X (it's the solution, the primal variable, and also the lagrangian multiplier in dual problem)
      X -= gamma * beta * (Z - AtR);
      //A.forward(X, AX);

      dX = X - X0;
      X0 = X;

      RdX = (niter==0) ? 1. : (dX).matrix().norm() / X0.matrix().norm();
      primal_gap = 0; //(AX + mu * R - Y).matrix().norm();
      dual_gap = (AtR - Z).matrix().norm();
      delta_gap = (Y * R).sum() - mu * (R * R).sum() - X.abs().sum();
      gap = fmax(fmax(primal_gap, dual_gap), delta_gap);

      // Print convergence information
      if (verbose) {	
	//res = (AX-Y).matrix().norm();
	res = mu * R.matrix().norm();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);
	printf("Iteration : %d\tRdX = %1.5e\tres = %1.5e\tgap = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\n", niter, RdX, res, gap, nL1, nL0/X.size());
      }
      niter++;
    }

    if (RdX > tol and gap > 1e-4 and verbose)
      cout<<"L1PADM terminates without convergence."<<endl;      

    nL1 = X.abs().sum();
    res = mu * R.matrix().norm();
    //res = (AX-Y).matrix().norm();
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
