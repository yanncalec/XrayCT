#include "SpAlgo.hpp"
// L1 minimization by Primal Alternating Direction Method
// This algorithm is very sensible to the choice of beta, smaller beta is, stronger the shrinkage effect
// To guarantee the convergence, tau* rho + gamma<2, rho is the maximum eigen value of (A^t A)
// It's strongly recommended to normalize A, st the maximum eigen value of (A^t A) is smaller than 1, then taking tau=gamma=1 guarantees convergence

namespace SpAlgo {
  ConvMsg L1PADM(LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X, ArrayXd &Z, ArrayXd &R,
		 double delta, double mu, double beta, double tau, double gamma,
		 double tol, int maxIter, double stol, double sparsity, int verbose, int Freq)
  {
    // A : system operator
    // W : weight for L1 norm
    // Y : data in vector form
    // X : initialization in vector form, output is rewritten on X

    // mu : L1 fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // delta : noise level constraint |Ax-b|<=delta, <0 for normal l1 model, ==0 for bpeq, >0 for bpdn
    // beta : data fidelity penalty, less important than mu, could be taken in same order as mu
    // tau : proximal penalty
    // gamma : Lagrangian step
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message
    // Freq:		// Check the support changement and print message every Freq iterations.
 
    bool dn = (delta > 0);		// true for constraint BPDN model : min |x|_1 st |Ax-b|_2 <= delta
    bool eq = (delta == 0); 		// true for eq BP model: min |x|_1 st Ax=b
    bool l1 = (delta < 0);		// true for l1 model: min |Ax-b|^2 + mu*|x|_1

    assert(beta>0 and tau>0 and gamma>0); // beta is used in any case
    if (l1)			// mu is used only for l1(non constraint) case
      assert(mu>0);

    if (verbose) {
      cout<<"-----L1 minimization by PADM-----"<<endl;
      if(l1)
	cout<<"Solve min |Ax-b|^2 + mu*|x|_1"<<endl;
      else if(eq)
	cout<<"Solve min |x|_1 st Ax=b"<<endl;
      else 
	cout<<"Solve min |x|_1 st |Ax-b|<=delta"<<endl;

      cout<<endl<<"Parameters :"<<endl;
      cout<<"Equality constraint : "<<eq<<endl;
      cout<<"Noise level : "<<delta<<endl;
      if (delta < 0)
	cout<<"L1 fidelity penalty : "<<mu<<endl;
      cout<<"Penalty beta : "<<beta<<endl;
      cout<<"Proximal penalty : "<<tau<<endl;
      cout<<"Lagrangian step : "<<gamma<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    //BlobImage *BI = dynamic_cast<BlobImage *>(const_cast<LinOp *>(&A));

    int sFreq = (int)fmax(Freq/5., 1);		// Check the support change every sFreq iterations.

    ArrayXd X0, dX;
    X0 = X;			

    // ArrayXd Z;			// Lagrangian multiplier
    // ArrayXd R;			// Residual or the auxilary variable
    // Z.setZero(Y.size()); 
    // R.setZero(Y.size()); 

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
   
    ArrayXi XSupp;
    ArrayXi XSupp0 = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
    int Xdiff = X.size();
    double rXdiff=1.;		// Relative size of support change

    ArrayXi MSupp;		// Main support (nnz-largest coeffs)
    MSupp.setZero(X.size());	
    ArrayXi MSupp0 = XSupp0;
    int Mdiff = X.size();
    double rMdiff=1.;		// Relative size of main support change
    const int nnz = (int)ceil(X.size() * sparsity);

    bool converged = false;

    //    while (niter < maxIter and RdX > tol and gap > 1e-4) {    
    while (niter < maxIter and not converged) {    
      // Update residual
      if (!eq) {
	if (dn)
	  R = l2_ball_projection(Z/beta - (AX - Y), delta);
	else
	  R = (Z/beta - (AX - Y)) * beta / (1./mu + beta); 
      }
      else {
	R.setZero();
      }

      // Update X by proximal step
      // Proximal step : <grad, x-x^k> + 1/2/tau * |x-x^k|^2
      A.backward(AX + R - Y - Z/beta, grad);

      // Print the message to observe the effect of beta when shit happens
      // toto = X - tau * grad;
      // cout<<toto.minCoeff()<<", "<<toto.maxCoeff()<<", "<<tau/beta<<endl;
      
      X = l1_shrink(X - tau * grad, tau / beta, W);
      A.forward(X, AX);

      // Update lagrangian multiplier Z
      Z -= gamma * beta * (AX + R - Y);

      dX = X - X0;
      X0 = X;

      RdX = (niter==0) ? 1. : (dX).matrix().norm() / X0.matrix().norm();
      gap = (AX + R - Y).matrix().norm();

      if (niter % sFreq == 0) {
	// Support changement of X
	if (stol>0) {
	  XSupp = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
	  Xdiff = (XSupp0 - XSupp).abs().sum();
	  XSupp0 = XSupp;
	  rXdiff = Xdiff*1./X.size();
	  converged = rXdiff<stol;
	}

	// Main support changement of X
	// if (BI != NULL)
	//   MSupp = BI->prod_mask(X, nz);
	// else
	if (sparsity > 0) {
	  MSupp = SpAlgo::NApprx_support(X, nnz);
	  Mdiff = (MSupp0 - MSupp).abs().sum();
	  MSupp0 = MSupp;
	  rMdiff = Mdiff*1./X.size();
	  // cout<<MSupp.minCoeff()<<", "<<MSupp.maxCoeff()<<endl;
	  // cout<<"MSupp.norm0 = "<<MSupp.abs().sum()<<" MSupp0+MSupp = "<<(MSupp0+MSupp).sum()<<endl;
	}
      }

      converged = converged or (RdX < tol);

      // Print convergence information
      if (verbose and niter % Freq == 0) {	
	res = (AX-Y).matrix().norm();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);

	printf("Iteration : %d\tRdX = %1.2e\t|AX-Y| = %1.5e\tgap = %1.2e\tL1 norm = %1.2e\tnon zero per. = %1.2e\tXdiff=%d\trXdiff=%1.2e\tMdiff=%d\trMdiff=%1.2e\n", niter, RdX, res, gap, nL1, nL0/X.size(), Xdiff, rXdiff, Mdiff, rMdiff);
      }
      niter++;
    }

    if (verbose>0 and not converged)
      //    if (RdX > tol and gap > 1e-4 and verbose)
      cout<<"L1PADM terminated without convergence."<<endl;      

    nL1 = X.abs().sum();
    res = (AX-Y).matrix().norm();
    if (dn)
      vobj = nL1;
    else
      vobj = 0.5 * res * res + mu * nL1;

    return ConvMsg(niter, vobj, res, nL1);
  }
}

