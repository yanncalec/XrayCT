#include "SpAlgo.hpp"
// TVAL3 steepest descent (gradient) implementation
// Nonnegativity is achieved by projected gradient.

namespace SpAlgo {
  ConvMsg TVAL3(const LinOp &A, const LinOp &G, const ArrayXd &Y, 
		ArrayXd &X, ArrayXd &Nu, ArrayXd &Lambda,
		bool tveq, bool aniso, bool nonneg, double mu, double beta,
		double tol_Inn, double tol_Out, size_t maxIter,
		int verbose, const ArrayXd *ref,
		void (*imsave)(const ArrayXXd &, const string &),
		const BlobImage *BI, const Array2i *dimObj, const string *outpath)
  {
    // A : system operator
    // G : gradient like operator
    // Y : data in vector form
    // X : initialization in vector form, output is rewritten on X
    // Nu : initial value of lagrange multiplier for GX = U
    // Lambda : initial value of lagrange multiplier for AX = Y constraint
    // tveq : true for TV equality model, false for TVL2 model
    // aniso : true for anisotrope (l1) TV norm
    // mu : data fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // beta : gradient fidelity penalty, less important than mu, could be taken in same order as mu
    // nonneg : true for positivity constraint
    // tol_Inn : inner iteration tolerance
    // tol_Out : outer iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message
    // ref : reference image
    // imsave : function handel for saving image (see CmdTools::imsave)
    // BI : BlobImage object with blob2pixel method, this is useful for saving intermediate results into a gif image
    // dimObj : desired screen image dimension
    // outpath : output path name for intermediate images

    if (verbose) {
      cout<<"-----Total Variation minimization by AL3 Method-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Positive constaint : "<<nonneg<<endl;
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      cout<<"Data fidelity penalty : "<<mu<<endl;
      cout<<"Gradient fidelty penalty : "<<beta<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Inner stopping tolerance : "<<tol_Inn<<endl;
      cout<<"Outer stopping tolerance : "<<tol_Out<<endl;
      cout<<"TV with equality constraint :"<<tveq<<endl;
    }
    //assert(mu>0 && beta>0);

    ArrayXd X0, X1;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd U, U0, U1;			// Auxiliary variable GX = U
  
    ArrayXd gradX, dgradX;	// Gradient of AL function wrt X
    ArrayXd AgradX, GgradX;	
    ArrayXd AX, GX;		

    double alpha, ngradX;
    ArrayXd AXmYmL, GXmUmN;
    ArrayXd H, Z, AH, GH;

    double RdX_Out = 1.; // Relative change in X, outer iteration stopping criteria
    double RdX_Inn = 1.; // Relative change in X, inner iteration stopping criteria
    size_t niter = 0;	 // Counter for iteration
    size_t gradRank = G.shapeY[1]; // dimension of gradient vector at each site

    double res, nTV;

    ConvMsg msg(maxIter);	// Convergent message
    double nY = Y.matrix().norm();	// norm of Y

    // Initialization
    AX = A.forward(X);  
    GX = G.forward(X);

    // temporary variables are noted by XX1 in inner loop and XX0 in outer loop.
    X0 = X; U0 = U;
    X1 = X; U1 = U;	 

    while (niter < maxIter and RdX_Out > tol_Out) {
 
      // Given new X, solve U by shrinkage
      if (aniso)
	U = TVTools::l1_shrink(GX - Nu/beta, beta);
      else
	U = TVTools::l2_shrink(GX - Nu/beta, beta, gradRank);

      // Given U, compute X by one-step descent method

      // Update temporary variables (Lambda, Nu modified at outer iteration)
      if (tveq)
	AXmYmL = AX-Y-Lambda/mu;
      else
	AXmYmL = AX-Y;

      GXmUmN = GX-U-Nu/beta;

      // Gradient wrt X of AL function
      gradX = mu * A.backward(AXmYmL) + beta * G.backward(GXmUmN);
      ngradX = (gradX *gradX).sum();

      // Steepest descent
      AgradX = A.forward(gradX);
      GgradX = G.forward(gradX);      
      alpha = ngradX / (mu*(AgradX*AgradX).sum() + beta*(GgradX*GgradX).sum());

      Z = X - alpha * gradX;
      if (nonneg) { // Projection gradient method
      	Z = (Z<0).select(0, Z);
      }

      H = Z - X;	// Decreasing vector      
      AH = A.forward(H);
      GH = G.forward(H);

      // Backtrack line search
      double tau = 1.;
      size_t citer = 1;
      double val = (AXmYmL+tau*AH).matrix().squaredNorm() * mu/2 + 
      	(GXmUmN+tau*GH).matrix().squaredNorm() * beta/2;
      double val0 = (AXmYmL).matrix().squaredNorm() * mu/2 + 
      	(GXmUmN).matrix().squaredNorm() * beta/2;
      double cst = (H * gradX).sum();
      
      while(val > val0 + 1e-3 * tau * cst and citer < 10) {
      	tau *= 0.5;
      	val = (AXmYmL+tau*AH).matrix().squaredNorm() * mu/2 + 
      	  (GXmUmN+tau*GH).matrix().squaredNorm() * beta/2;
      	citer++;
      }
      
      if (citer >= 10 and verbose) 
      	cout<<"Backtrack line search failed."<<endl;

      H = Z - X;	// Decreasing vector      
      AH = A.forward(H);
      GH = G.forward(H);

      X += H;
      AX += tau * AH;
      GX += tau * GH;

      // Update inner states of X and U
      RdX_Inn = (X - X1).matrix().norm() / X1.matrix().norm();
      X1 = X; U1 = U;

      // Update multipliers and enter in next outer iteration when
      // rel.changement of inner iteration is small
      if (RdX_Inn <= tol_Inn) {
	Nu -= beta * (GX - U);
	if (tveq)  // Equality constraint
	  Lambda -= mu * (AX - Y);	

	// Save convergence information
	RdX_Out = (X - X0).matrix().norm() / X0.matrix().norm();
	msg.RelErr.push_back(RdX_Out);
	if (ref != 0) {
	  msg.SNR.push_back(Tools::SNR(X, *ref));
	  msg.Corr.push_back(Tools::Corr(X, *ref));
	}
	// Update outer states of X and U
	X0 = X;
	U0 = U;   
    
	// Print convergence information
	if (verbose) {	
	  nTV = TVTools::GTVNorm(GX, gradRank) / G.shapeY[0];
	  res = (AX-Y).matrix().norm() / nY;
      
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
