#include "SpAlgo.hpp"

namespace SpAlgo {
  // Total Variation minimization by Inexact Alternating Direction Method    
  ConvMsg TVIADM(const LinOp &A, const LinOp &G, const LinOp &F, 
		 const ArrayXd &Y, ArrayXd &X0, ArrayXd &Nu, ArrayXd &Lambda,
		 bool tveq, bool aniso, bool nonneg, double mu, double beta, double tau,
		 double tol, size_t maxIter, int verbose, const ArrayXd *ref,
		 void (*imsave)(const ArrayXXd &, const string &),
		 const BlobImage *BI, const Array2i *dimObj, const string *outpath)
  {
    // A : system operator
    // G : gradient like operator
    // F : FFT solver for symmetric system of type (G^*G + cst) X = Z
    // Y : data in vector form
    // X0 : initialization in vector form, output is rewritten on X
    // Nu : initial value of lagrange multiplier for GX = U
    // Lambda : initial value of lagrange multiplier for AX = Y constraint
    // tveq : true for TV equality model, false for TVL2 model
    // aniso : true for anisotrope (l1) TV norm
    // nonneg : true for positivity constraint
    // mu : data fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // beta : gradient fidelity penalty, less important than mu
    // tau : proximal term penalty
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message
    // ref : reference image
    // imsave : function handel for saving image (see CmdTools::imsave)
    // BI : BlobImage object with blob2pixel method, this is useful for saving intermediate results into a gif image
    // dimObj : desired screen image dimension
    // outpath : output path name for intermediate images

    //assert(mu>0 && beta>0);
    if (verbose) {
      cout<<"-----Total Variation minimization by Inexact Alternating Direction Method-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"TV with equality constraint : "<<tveq<<endl;
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      cout<<"Positive constaint : "<<nonneg<<endl;
      cout<<"Data fidelity penalty : "<<mu<<endl;
      cout<<"Gradient fidelty penalty : "<<beta<<endl;
      cout<<"Proximal fidelty penalty : "<<tau<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
    }

    ArrayXd X = X0;		// Unknown
    ArrayXd U, U0;		// Auxiliary variable GX = U
    ArrayXd gk;

    double RdX = 1.; // Relative change in X, outer iteration stopping criteria
    double res = 1.; // Relative residual of data 
    double nY = Y.matrix().norm();	// norm of Y
    double nTV;				// TV norm of X

    size_t niter = 0;	 // Counter for iteration
    size_t gradRank = G.shapeY[1]; // dimension of gradient vector at each site
    //cout<<"gradRank = "<<gradRank<<endl;

    ConvMsg msg(maxIter);	// Convergent message

    // Initialization of temporary variables
    ArrayXd AXmY = A.forward(X) - Y;  
    ArrayXd GX = G.forward(X);
    // Given X, solve U by shrinkage
    if (aniso)
      U = TVTools::l1_shrink(GX - Nu/beta, beta);
    else
      U = TVTools::l2_shrink(GX - Nu/beta, beta, gradRank);

    while (niter < maxIter and RdX > tol) {
      // Given U, compute X by FFT
      // Solve normal equation
      if (tveq)
	gk = A.backward(AXmY-Lambda/mu);
      else
	gk = A.backward(AXmY);

      X = F.forward(mu/beta/tau * (X - tau * gk) + G.backward(Nu/beta + U));
      // Positive constraint prevents the convergence!!
      if (nonneg)
      	X = (X<0).select(0, X);

      // Update tempo variables
      GX = G.forward(X);
      AXmY = A.forward(X) - Y;

      // Given X, solve U by shrinkage
      if (aniso)
	U = TVTools::l1_shrink(GX - Nu/beta, beta);
      else
	U = TVTools::l2_shrink(GX - Nu/beta, beta, gradRank);

      // Update multipliers
      Nu -= beta * (GX - U);
      if (tveq)	// Equality constraint
      	Lambda -= mu * AXmY;

      // Save convergence information
      RdX = (X - X0).matrix().norm() / X0.matrix().norm();

      // Update X and U
      X0 = X;
      U0 = U;  

      // Save & print convergence information
      msg.RelErr.push_back(RdX);
      if (ref != 0) {
	msg.SNR.push_back(Tools::SNR(X, *ref));
	msg.Corr.push_back(Tools::Corr(X, *ref));
      }
        
      if (verbose and niter % 10 == 0) {	// Register every 10 image
	nTV = TVTools::GTVNorm(GX, gradRank).sum() / G.shapeY[0];
	res = (AXmY).matrix().norm() / nY;

	if (ref == 0) 
	  printf("Iteration : %lu \tRdx = %3.5f\tres. = %3.5f\tTV norm = %f\n", niter, RdX, res, nTV);
	//cout<<"Iteration "<<niter<<"\ttol = "<<msg.RelErr.back()<<"\tres. = "<<res<<"\tTV norm = "<<nTV<<endl;
	else
	  printf("Iteration : %lu \tRdx = %3.5f\tCorr = %3.5f\t SNR = %3.5f\tres. = %3.5f\tTV norm = %f\n", niter, RdX, msg.Corr.back(), msg.SNR.back(), res, nTV);
	//cout<<"Iteration "<<niter<<"\ttol = "<<msg.RelErr.back()<<"\tCorr = "<<msg.Corr.back()<<"\tSNR ="<<msg.SNR.back()<<"\tres. = "<<res<<"\tTV norm = "<<nTV<<endl;
	if (BI != 0 and imsave != 0 and outpath != 0 and dimObj != 0) {
	  string fname = *outpath + "/iter_" + Tools::itoa_fixlen3(niter) + ".png";	      
	  ArrayXXd imr = (*BI).blob2pixel(X, *dimObj);
	  (*imsave)(imr, fname);
	}
      }
      niter++;
    }
    return msg;
  }
}
