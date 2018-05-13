#include "SpAlgo.hpp"
// L1 minimization by Iterative Soft-Thresholding

namespace SpAlgo {
  ConvMsg L0IHT_Nrml(const LinOp &A, const ArrayXd &Y, ArrayXd &X, 
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
      cout<<"-----L0 minimization by Normalized Hard Iterative Thresholding Method-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"L0 constraint : "<<nterm<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    // const gsl_rng_type * Trng;
    // gsl_rng *rng;     
    // gsl_rng_env_setup();     
    // Trng = gsl_rng_default;
    // rng = gsl_rng_alloc (Trng);

    X.setZero();
    ArrayXi S = NApprx_support(A.backward(Y), nterm);
    ArrayXi S1;
    ArrayXd gradX, sgradX, AsgradX;
    gradX.setZero(X.size());
    sgradX.setZero(X.size());
    AsgradX.setZero(Y.size());

    ArrayXd Z, dX, AdX;
    AdX.setZero(Y.size());

    ArrayXd AX; AX.setZero(Y.size());
    A.forward(X, AX);

    double RdX = 1.;
    double ndX, nAdX, res, nL0, nL1;
    int niter = 0;
    double cst = 1e-2;
    double gradstep;

    while (niter < maxIter and RdX > tol) {    
      A.backward(AX - Y, gradX);
      sgradX = (S == 0).select(0, gradX);
      A.forward(sgradX, AsgradX);
      gradstep = sgradX.matrix().squaredNorm() / AsgradX.matrix().squaredNorm();
      
      Z = NApprx(X - gradstep * gradX, nterm);
      S1 = NApprx_support(Z, nterm);
      dX = Z-X;
      ndX = dX.matrix().squaredNorm();

      if ((S1 - S).abs().sum() == 0)
	X = Z;
      else {	
	A.forward(dX, AdX);
	nAdX = AdX.matrix().squaredNorm();
	while (gradstep > (1-cst) * ndX / nAdX) {
	  gradstep /= 2;
	  Z = NApprx(X - gradstep * gradX, nterm);
	  dX = Z-X;
	  ndX = dX.matrix().squaredNorm();
	  A.forward(dX, AdX);
	  nAdX = AdX.matrix().squaredNorm();	  
	}
	X = Z;	  
      }

      A.forward(X, AX);
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
