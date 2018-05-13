#include "SpAlgo.hpp"
// TVL0AL3 BB implementation
// Not stable

namespace SpAlgo {
  ConvMsg TVL0AL3(const LinOp &A, const LinOp &G, const ArrayXd &W, const ArrayXd &Y,
		ArrayXd &X, ArrayXd &Nu, ArrayXd &Lambda,
		bool eq, bool aniso, bool nonneg, int nterm,
		double mu, double beta,
		double tol_Inn, double tol_Out, int maxIter, int verbose)
  {

    if (verbose) {
      cout<<"-----Total Variation minimization by AL3 Method-----"<<endl;
      cout<<"Solve min 1/2 |Ax-b|^2 + mu*|x|_TV with l0 constraint"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Positive constaint : "<<nonneg<<endl;
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      cout<<"L0 constraint : "<<nterm<<endl;
      cout<<"TV multiplier : "<<mu<<endl;
      cout<<"Gradient fidelty penalty : "<<beta<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Inner stopping tolerance : "<<tol_Inn<<endl;
      cout<<"Outer stopping tolerance : "<<tol_Out<<endl;
      cout<<"TV with equality constraint :"<<eq<<endl;
    }

    ArrayXd X0, X1, dX;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd Xabs;			// abs value of X
    ArrayXd U;			// Auxiliary variable GX = U
  
    ArrayXd gradX, gradX1, dgradX;	// Gradient of AL function wrt X
    ArrayXd AgradX(A.get_dimY()), GgradX(G.get_dimY());	
    ArrayXd AX, GX;		

    ArrayXd totoX1, totoX2;
    totoX1.setZero(X.size());
    totoX2.setZero(X.size());

    double gradstep;		// gradient step length
    ArrayXd AXmYmL, GXmUmN;

    double RdX_Out = 1.; // Relative change in X, outer iteration stopping criteria
    double RdX_Inn = 1.; // Relative change in X, inner iteration stopping criteria
    int niter = 0;	 // Counter for iteration
    int niter_Out = 0;	 // Counter for outer iteration

    int gradRank = G.get_shapeY().x(); // dimension of gradient vector at each site
    //cout<<"gradRank="<<gradRank<<endl;

    double res; // Relative residual of data 
    double nL0, nL1, nTV;				// L0, L1 and TV norm of X

    double nY = Y.matrix().norm();	// norm of Y
    double vobj = 0;			// Value of objective function

    ArrayXi Supp;
    ArrayXi Supp0 = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
    int Sdiff = 0;

    // Initialization
    AX = A.forward(X);  
    GX = G.forward(X);
    X0 = X; X1 = X;

    while (niter < maxIter and RdX_Out > tol_Out) {    
      // Given new X, solve U by shrinkage
      // Solve min mu/beta*\sum_k W_k*|U_k| + 1/2*|GX-Nu/beta-U|^2
      if (aniso)
	U = l1_shrink(GX - Nu/beta, mu/beta, W);
      else
	U = l2_shrike(GX - Nu/beta, mu/beta, gradRank, W);

      // Given U, compute X by one-step descent method
      // Update temporary variables first (Lambda, Nu modified at outer iteration)
      if (eq)
	AXmYmL = AX-Y-Lambda;
      else
	AXmYmL = AX-Y;
      GXmUmN = GX-U-Nu/beta;

      // Gradient wrt X of AL function
      //gradX = mu * A.backward(AXmYmL) + beta * G.backward(GXmUmN);
      A.backward(AXmYmL, totoX1);
      G.backward(GXmUmN, totoX2);
      gradX = totoX1 + beta * totoX2;

      if (niter == 0) { // Do steepest descent at 1st iteration
	// Gradient of AL function      
	A.forward(gradX, AgradX);
	G.forward(gradX, GgradX);      
	gradstep = (gradX*gradX).sum() / ((AgradX*AgradX).sum() + beta*(GgradX*GgradX).sum());
      }
      else {
	// Set gradstep through BB formula
	dgradX = gradX - gradX1;
	gradstep = (dX * dX).sum() / (dX * dgradX).sum();
      }
      
      X -= gradstep * gradX;

      if (nonneg) // Projection 
	X = (X<0).select(0, X);      

      // Keep only the biggest N-terms
      if (nterm > 0 and nterm < X.size()) {
	Xabs = X.abs();
	double val = Tools::Nth_maxCoeff(Xabs.data(), X.size(), nterm);
	X = (Xabs<val).select(0, X);
	Supp = (Xabs<val).select(0, ArrayXi::Ones(X.size()));
	Sdiff = (Supp0 - Supp).abs().sum();
	Supp0 = Supp;
      }

      A.forward(X, AX);
      G.forward(X, GX);
      dX = X - X1;

      // Update inner states of X and U
      RdX_Inn = dX.matrix().norm() / X1.matrix().norm();
      X1 = X;
      gradX1 = gradX;

      // Update multipliers and enter in next outer iteration when
      // rel.changement of inner iteration is small
      if (RdX_Inn <= tol_Inn) {
	Nu -= beta * (GX - U);
	if (eq)  // Equality constraint
	  Lambda -= AX - Y;	

	// Save convergence information
	RdX_Out = (niter_Out == 0) ? 1. : (X - X0).matrix().norm() / X0.matrix().norm();

	// Update outer states of X and U
	X0 = X;
    
	// Print convergence information
	if (verbose) {	
	  res = (AX - Y).matrix().norm();      
	  nTV = GTVNorm(GX, gradRank).sum();
	  vobj = 0.5 * res*res + mu*nTV;
	  nL1 = X.abs().sum();
	  nL0 = Tools::l0norm(X);

	  printf("Iteration : %d\ttol_Out = %1.5e\tvobj = %1.5e\tres = %1.5e\trres = %1.5e\tTV norm = %1.5e\trTV norm = %1.5e\tL1 norm = %1.5e\tnon zero per. = %2.2f\tSdiff = %d\n", niter, RdX_Out, vobj, res, res/nY, nTV, nTV/G.get_shapeY().y(), nL1, nL0*1./X.size(), Sdiff);

	}
	niter_Out++;
      }
      niter++;
    }

    if (niter >= maxIter and verbose)
      cout<<"TVL0AL3 terminates without convergence."<<endl;
    
    res = (AX - Y).matrix().norm();
    nTV = GTVNorm(GX, gradRank).sum();
    vobj = 0.5 * res*res + mu*nTV;
    return ConvMsg(niter, vobj, res, nTV);
  }
}

    // Following is for N-term apprx, preparation of random number generator
    // const gsl_rng_type * Trng;
    // gsl_rng *rng;     
    // gsl_rng_env_setup();     
    // Trng = gsl_rng_default;
    // rng = gsl_rng_alloc (Trng);
    // end

	  // if (verbose > 1 and BI != 0 and imsave != 0 and outpath != 0 and dimObj != 0) {	      
	  //   string fname = *outpath + "/iter_" + Tools::itoa_fixlen3(niter);	      
	  //   ArrayXXd imr = BI->blob2pixel(X, *dimObj);
	  //   (*imsave)(imr, fname, false);
	  // }
