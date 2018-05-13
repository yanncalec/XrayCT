#include "SpAlgo.hpp"
// TV-L1 minimization by splitting algorithm
// Reference : "An Efficient ALgorithm for Compressed MR Imaging using Total Variation and Wavelets"
// Unlike TVL1Prxm, this method solves only the unconstrainted optimization problem.

namespace SpAlgo {
  ConvMsg TVL1SPL(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X, 
		  bool aniso, double alpha, double beta, double tau1, double tau2,
		  double tol, int maxIter,  double stol, double sparsity, int verbose)
  {
    // A : system operator
    // G : gradient like operator
    // Y : data in vector form
    // X : initialization in vector form, output is rewritten on X
    // aniso : true for anisotrope (l1) TV norm
    // alpha : L1 fidelity penalty
    // beta : gradient fidelity penalty, less important than mu, could be taken in same order as mu
    // tol_Out : outer iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    if (verbose) {
      cout<<"-----Total Variation L1 minimization by Operator Splitting Method-----"<<endl;
      cout<<"Solve min 1/2 |Ax-b|^2 + alpha*|x|_TV + beta*|x|_L1"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      cout<<"TV penalty : "<<alpha<<endl;
      cout<<"L1 penalty : "<<beta<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Outer stopping tolerance : "<<tol<<endl;
    }

    int sFreq = 2;		// Check the support change every sFreq iterations.
    int Freq = 2;		// print message every Freq iterations.

    ArrayXd X0, dX;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd U;			// Auxiliary variable GX = U
    ArrayXd S, T, S0, T0;		// Auxiliary variables
    S.setZero(X.size()); S0 = S;
    T.setZero(G.get_dimY()); T0 = T;

    ArrayXd gradX;	// Gradient of AL function wrt X
    ArrayXd AX, GX;		

    ArrayXd totoX1(X.size()), totoX2(X.size());

    double RdX = 1.; // Relative change in X, inner iteration stopping criteria
    int niter = 0;	 // Counter for iteration

    int gradRank = G.get_shapeY().x(); // dimension of gradient vector at each site
    //cout<<"gradRank="<<gradRank<<endl;

    double res; // Relative residual of data 
    double nL1, nL0, nTV;	// L1, L0, TV norm of X

    double nY = Y.matrix().norm();	// norm of Y
    double vobj = 0;			// Value of objective function

    ArrayXi XSupp;		// Support of X
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

    // Initialization
    AX = A.forward(X);  
    GX = G.forward(X);
    U.setZero(GX.size());
    X0 = X;

    while (niter < maxIter and not converged) {    
      // Step a. set S
      A.backward(AX-Y, totoX1);
      G.backward(U, totoX2);
      S = X - tau1 * (totoX1 + alpha * totoX2);

      // Step b. set T
      T = U + tau2 * GX;

      // Step c. set X
      X = l1_shrink(S, tau1*beta);
      
      // Step d. set U
      if (aniso)
	U = l1_shrink(T, 1/tau2);
      else
	U = l2_shrike(T, 1/tau2, gradRank);

      //S0 = S; T0 = T;
      A.forward(X, AX);
      G.forward(X, GX);
      dX = X - X0;

      // Update inner states of X and U
      RdX = (niter == 0) ? 1. : dX.matrix().norm() / X0.matrix().norm();
      X0 = X;

      if (niter % sFreq == 0) {
	// Support changement of X
	XSupp = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
	Xdiff = (XSupp0 - XSupp).abs().sum();
	XSupp0 = XSupp;
	rXdiff = Xdiff*1./X.size();
	converged = rXdiff<stol;

	if (sparsity > 0 and sparsity < 1) {
	  MSupp = SpAlgo::NApprx_support(X, nnz);
	  Mdiff = (MSupp0 - MSupp).abs().sum();
	  MSupp0 = MSupp;
	  rMdiff = Mdiff*1./X.size();
	  converged = converged or (rMdiff < fmin(stol, 5e-4));
	}
      }

      converged = converged or (RdX < tol);

      // Print convergence information
      if (verbose>0 and niter % Freq == 0) {	
	res = (AX-Y).matrix().norm();
	nTV = GTVNorm(GX, gradRank).sum();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);
	vobj = 0.5 * res*res + alpha*nTV + beta*nL1;

	if (sparsity > 0 and sparsity < 1)
	  printf("Iteration : %d\tRdX = %1.5e\tres = %1.5e\tL1 norm = %1.5e\tTV norm = %1.5e\tnon zero per. = %1.5e\tXdiff=%d\trXdiff=%1.5e\tMdiff=%d\trMdiff=%1.5e\n", niter, RdX, res,  nL1, nTV, nL0/X.size(), Xdiff, rXdiff, Mdiff, rMdiff);
	else
	  printf("Iteration : %d\tRdX = %1.5e\tres = %1.5e\tL1 norm = %1.5e\tTV norm = %1.5e\tnon zero per. = %1.5e\tXdiff=%d\trXdiff=%1.5e\n", niter, RdX, res, nL1, nTV, nL0/X.size(), Xdiff, rXdiff);
      }
      niter++;
    }
      
    if (niter >= maxIter and verbose)
      cout<<"TVL1SPL terminated without convergence."<<endl;
    
    res = (AX - Y).matrix().norm();
    nTV = GTVNorm(GX, gradRank).sum();
    nL1 = X.abs().sum();
    vobj = 0.5 * res*res + alpha*nTV + beta*nL1;

    return ConvMsg(niter, vobj, res, alpha*nTV + beta*nL1);
  }
}
