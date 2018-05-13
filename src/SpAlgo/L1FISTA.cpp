#include "SpAlgo.hpp"
#include "BlobProjector.hpp"
// L1 minimization by Fast Iterative Soft-Thresholding Algorithm (FISTA)

namespace SpAlgo {
  ConvMsg L1FISTA(LinOp &A, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X,
		  double mu, double tol, int maxIter, 
		  double stol, double sparsity, int verbose, int Freq,
		  void (*savearray)(const ArrayXd &, const string &), const string * outpath)
  {
    // A : system operator
    // Y : data in vector form
    // W : weight for L1 norm
    // X : initialization in vector form, output is rewritten on X
    // mu : L1 fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    assert(mu>0);

    //BlobProjector *P = dynamic_cast<BlobProjector *>(const_cast<LinOp *>(&A));
    if (verbose>0) {
      cout<<"-----L1 minimization by FISTA method with BB gradient step-----"<<endl;
      cout<<"Solve min |Ax-b|^2 + mu*|x|_1"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"L1 fidelity penalty : "<<mu<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
      cout<<"Main support sparsity : "<<sparsity<<endl;
      cout<<"Main support chg. stopping tolerance : "<<stol<<endl;
    }

    BlobImage *BI = dynamic_cast<BlobImage *>(const_cast<LinOp *>(&A));
    // if (BI==NULL)
    //   cout<<"Conversion failed : LinOp -> BlobImage"<<endl;

    //int sFreq = (int)fmax(Freq/5., 2);		// Check the support change every sFreq iterations.
    const int sFreq = 5;		// Check the support change every sFreq iterations.

    ArrayXd X0, dX, dX0, Z, Z0, dZ;
    X0 = X;			
    Z = X; 
    Z0 = X;
  
    ArrayXd AX = A.forward(X);
    ArrayXd AX0 = AX;
    ArrayXd AZ = AX;

    ArrayXd gradZ, gradZ0, dgradZ;
    gradZ.setZero(X.size());
    gradZ0.setZero(X.size());
    dgradZ.setZero(X.size());

    double gradstep;		// BB gradient step
    double RdX = 1., RdX0=1.; // Relative change in X, iteration stopping criteria
    int niter = 0;	 // Counter for iteration

    double res;			   // Residual of data 
    double nL1;			   // L1 norm of X
    double nL0;			   // L0 norm of X
    double nY = Y.matrix().norm(); // norm of Y
    double vobj;		   // value of objective function 1/2|Ax-y|^2 + |x|_1
    double tau = 1., tau0=1.;
    const int nnz = (int)ceil(X.size() * sparsity);

    ArrayXi XSupp;		// Support of X
    XSupp.setZero(X.size());
    ArrayXi XSupp0 = XSupp; //(X.abs()>0).select(1, ArrayXi::Zero(X.size()));
    int Xdiff = X.size();
    double rXdiff=1.;		// Relative size of support change

    ArrayXi MSupp;		// Main support (nnz-largest coeffs)
    MSupp.setZero(X.size());	
    ArrayXi MSupp0 = XSupp0;
    int Mdiff = X.size();
    double rMdiff=1.;		// Relative size of main support change

    bool converged = false;

    //while (niter < maxIter and RdX > tol and (!debias or Mdiff > 0)) {    
    while (niter < maxIter and not converged) {    
      A.backward(AZ-Y, gradZ);

      if (niter == 0) { // Do steepest descent at 1st iteration
      	// Gradient of AL function      
      	ArrayXd AgradZ = A.forward(gradZ);
      	//cout<<gradZ.matrix().norm()<<", "<<AgradZ.matrix().norm()<<endl;
      	gradstep = (gradZ*gradZ).sum() / (AgradZ*AgradZ).sum();
      }
      else {
      	// Set gradstep through BB formula
      	dgradZ = gradZ - gradZ0;
      	gradstep = (dZ * dZ).sum() / (dZ * dgradZ).sum();
      }
      
      X = l1_shrink(Z - gradstep*gradZ, gradstep*mu, W);

      A.forward(X, AX);
      dX = X - X0;
      tau = (1+sqrt(1+4*tau*tau))/2;
      Z = X + (tau0 - 1)/tau * dX;
      dZ = Z - Z0;
      
      AZ = (1+(tau0 - 1)/tau)*AX - (tau0-1)/tau * AX0;

      X0 = X;
      Z0 = Z;
      AX0 = AX;
      gradZ0 = gradZ;
      tau0 = tau;

      RdX = (niter==0) ? 1. : (dX).matrix().norm() / X0.matrix().norm();
      //RddX = (niter<2) ? 1. : (dX-dX0).matrix().norm() / dX0.matrix().norm();
      dX0 = dX;

      if (niter % sFreq == 0) {
	// Support changement of X
	XSupp = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
	Xdiff = (XSupp0 - XSupp).abs().sum();
	XSupp0 = XSupp;
	rXdiff = Xdiff*1./X.size();
	converged = converged or (rXdiff<stol);

	if (sparsity > 0 and sparsity < 1) {
	  // Two ways to calculate the main support of X
	  // 1. By N-term approximation
	  MSupp = SpAlgo::NApprx_support(X, nnz);
	  // 2. By N-term approximation on the scale product. Numerical exp show that this method has no advantage.
	  // MSupp = BI->prodmask(X, 0, 1); 
	  Mdiff = (MSupp0 - MSupp).abs().sum();
	  MSupp0 = MSupp;
	  rMdiff = Mdiff*1./X.size();
	  converged = converged or (rMdiff < fmin(stol, 5e-4));
	}
      }

      converged = converged or (RdX < tol and RdX0 < tol); // Must have two consequtive small values
      RdX0 = RdX;

      if (niter % Freq == 0 and savearray != NULL and outpath != NULL){
	char buffer[256];
	sprintf(buffer, "%s/tempo_%d", outpath->c_str(), niter);
	savearray(X, buffer);
      }

      // Print convergence information
      if (verbose>0 and niter % Freq == 0) {	
	res = (AX-Y).matrix().norm();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);

	printf("Iteration : %d\tRdX = %1.2e\t|AX-Y| = %1.5e\tL1 norm = %1.2e\tnon zero per. = %1.2e\tXdiff=%d\trXdiff=%1.2e\tMdiff=%d\trMdiff=%1.2e\n", niter, RdX, res,  nL1, nL0/X.size(), Xdiff, rXdiff, Mdiff, rMdiff);
	// if (sparsity > 0 and sparsity < 1)
	//   printf("Iteration : %d\tRdX = %1.2e\tRddX = %1.2e\tres = %1.2e\tL1 norm = %1.2e\tnon zero per. = %1.2e\tXdiff=%d\trXdiff=%1.2e\tMdiff=%d\trMdiff=%1.2e\n", niter, RdX, RddX, res,  nL1, nL0/X.size(), Xdiff, rXdiff, Mdiff, rMdiff);
	// else
	//   printf("Iteration : %d\tRdX = %1.2e\tRddX = %1.2e\tres = %1.2e\tL1 norm = %1.2e\tnon zero per. = %1.2e\tXdiff=%d\trXdiff=%1.2e\n", niter, RdX, RddX, res, nL1, nL0/X.size(), Xdiff, rXdiff);
      }
      niter++;
    }

    if (verbose>0 and not converged)
      cout<<"L1FISTA terminated without convergence."<<endl;

    nL1 = X.abs().sum();
    res = (AX-Y).matrix().norm();
    vobj = 0.5 * res * res + mu * nL1;
    return ConvMsg(niter, vobj, res, nL1);
  }
}


// The following is the IST algorithm (slower)

    // while (niter < maxIter and RdX > tol) {    
    //   A.backward(AX-Y, gradX);

    //   if (niter == 0) { // Do steepest descent at 1st iteration
    // 	// Gradient of AL function      
    // 	ArrayXd AgradX = A.forward(gradX);
    // 	gradstep = (gradX*gradX).sum() / (AgradX*AgradX).sum();
    //   }
    //   else {
    // 	// Set gradstep through BB formula
    // 	dgradX = gradX - gradX0;
    // 	gradstep = (dX * dX).sum() / (dX * dgradX).sum();
    //   }
      
    //   X = l1_shrink(X - gradstep * gradX, gradstep*mu, W);

    //   A.forward(X, AX);  
    //   dX = X - X0;
    //   X0 = X;
    //   gradX0 = gradX;

    //   RdX = (niter==0) ? 1. : (dX).matrix().norm() / X0.matrix().norm();
    // }
