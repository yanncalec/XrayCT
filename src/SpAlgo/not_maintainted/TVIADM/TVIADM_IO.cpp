#include "SpAlgo.hpp"
// Total Variation minimization by Inexact Alternating Direction Method    
// Inner-outer iteration implementation

namespace SpAlgo {
  ConvMsg TVIADM(const LinOp &A, const LinOp &G, const LinOp &F,
		 const ArrayXd &Y, ArrayXd &X, ArrayXd &Nu, ArrayXd &Lambda,
		 bool tveq, bool aniso, bool nonneg, double mu, double beta, double tau,
		 double tol_Inn, double tol_Out, size_t maxIter,
		 int verbose, const ArrayXd *ref,
		 void (*imsave)(const ArrayXXd &, const string &),
		 const BlobImage *BI, const Array2i *dimObj, const string *outpath)
  {
    // A : system operator
    // G : gradient like operator
    // F : FFT solver for symmetric system of type (G^*G + cst) X = Z
    // Y : data in vector form
    // X : initialization in vector form, output is rewritten on X
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
      cout<<"Inner stopping tolerance : "<<tol_Inn<<endl;
      cout<<"Outer stopping tolerance : "<<tol_Out<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
    }

    ArrayXd X0, X1, dX;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd U;			// Auxiliary variable GX = U
    ArrayXd gk;

    double RdX_Out = 1.; // Relative change in X, outer iteration stopping criteria
    double RdX_Inn = 1.; // Relative change in X, inner iteration stopping criteria

    double res; // Relative residual of data 
    double nTV;				// TV norm of X
    double nY = Y.matrix().norm();	// norm of Y

    size_t niter = 0;	 // Counter for iteration
    size_t gradRank = G.shapeY[1]; // dimension of gradient vector at each site
    //cout<<"gradRank = "<<gradRank<<endl;

    ConvMsg msg(maxIter);	// Convergent message

    // Initialization of temporary variables
    ArrayXd AXmY = A.forward(X) - Y;  
    ArrayXd GX = G.forward(X);

    X0 = X; X1 = X;

    while (niter < maxIter and RdX_Out > tol_Out) {
      // Given X, solve U by shrinkage
      if (aniso)
	U = TVTools::l1_shrink(GX - Nu/beta, beta);
      else
	U = TVTools::l2_shrink(GX - Nu/beta, beta, gradRank);

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
      dX = X - X1;

      // Update inner states of X and U
      RdX_Inn = dX.matrix().norm() / X1.matrix().norm();
      X1 = X;

      // Update multipliers and enter in next outer iteration when
      // rel.changement of inner iteration is small
      if (RdX_Inn <= tol_Inn) {
	Nu -= beta * (GX - U);
	if (tveq)  // Equality constraint
	  Lambda -= mu * AXmY;	

	// Save & print convergence information
	RdX_Out = (X - X0).matrix().norm() / X0.matrix().norm();
	msg.RelErr.push_back(RdX_Out);
	if (ref != 0) {
	  msg.SNR.push_back(Tools::SNR(X, *ref));
	  msg.Corr.push_back(Tools::Corr(X, *ref));
	}
	// Update outer states of X and U
	X0 = X;

	if (verbose and niter % 10 == 0) {	// Register every 10 image
	  nTV = TVTools::GTVNorm(GX, gradRank) / G.shapeY[0];
	  res = (AXmY).matrix().norm() / nY;

	  if (ref == 0)
	    printf("Iteration : %lu\ttol_Out = %2.5f\tres. = %2.5f\tTV norm = %f\n", niter, msg.RelErr.back(), res, nTV);
	  else
	    printf("Iteration : %lu\ttol_Out = %2.5f\tCorr = %2.5f\tSNR = %3.5f\tres. = %2.5f\tTV norm = %f\n", niter, msg.RelErr.back(), msg.Corr.back(), msg.SNR.back(), res, nTV);

	  if (BI != 0 and imsave != 0 and outpath != 0 and dimObj != 0) {
	    string fname = *outpath + "/iter_" + Tools::itoa_fixlen3(niter) + ".png";	      
	    ArrayXXd imr = (*BI).blob2pixel(X, *dimObj);
	    (*imsave)(imr, fname);
	  }
	}
      }
      niter++;
    }
    return msg;
  }
}
