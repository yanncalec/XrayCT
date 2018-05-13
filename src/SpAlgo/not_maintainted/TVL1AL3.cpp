#include "SpAlgo.hpp"
// TV-L1AL3 BB implementation

namespace SpAlgo {
  ConvMsg TVL1AL3(LinOp &A, LinOp &G, const ArrayXd &W, const ArrayXd &Y,
		  ArrayXd &X, ArrayXd &Nu, ArrayXd &Lambda, ArrayXd &Omega,
		  bool eq, bool aniso, double mu, double gamma, double alpha, double beta,
		  double tol_Inn, double tol_Out, int maxIter, int verbose)
  {
    // A : system operator
    // G : gradient like operator
    // Y : data in vector form
    // W : weight for TV norm
    // X : initialization in vector form, output is rewritten on X
    // Nu : initial value of lagrange multiplier for GX = U
    // Lambda : initial value of lagrange multiplier for AX = Y constraint
    // eq : true for TV equality model, false for TVL2 model
    // aniso : true for anisotrope (l1) TV norm
    // mu : data fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // gamma : weight for L1 penalty
    // alpha : L1 fidelity penalty
    // beta : gradient fidelity penalty, less important than mu, could be taken in same order as mu
    // tol_Inn : inner iteration tolerance
    // tol_Out : outer iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    if (verbose) {
      cout<<"-----Total Variation L1 minimization by AL3 Method-----"<<endl;
      cout<<"Solve min 1/2 |Ax-b|^2 + gamma*muTV*|x|_TV + (1-gamma)*muL1*|x|_L1"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      cout<<"Data fidelity : "<<mu<<endl;
      cout<<"TV to L1 weight : "<<gamma<<endl;
      cout<<"L1 fidelity penalty : "<<alpha<<endl;
      cout<<"Gradient fidelity penalty : "<<beta<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Inner stopping tolerance : "<<tol_Inn<<endl;
      cout<<"Outer stopping tolerance : "<<tol_Out<<endl;
      cout<<"TVL1 with equality constraint :"<<eq<<endl;
    }

    ArrayXd X0, X1, dX;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd U;			// Auxiliary variable GX = U
    ArrayXd V;			// Auxiliary variable X = V
  
    ArrayXd gradX, gradX1, dgradX;	// Gradient of AL function wrt X
    ArrayXd AgradX(A.get_dimY()), GgradX(G.get_dimY());	
    ArrayXd AX, GX;		

    ArrayXd totoX1(X.size()), totoX2(X.size());

    double gradstep;
    ArrayXd AXmYmL, GXmUmN;

    double RdX_Out = 1.; // Relative change in X, outer iteration stopping criteria
    double RdX_Inn = 1.; // Relative change in X, inner iteration stopping criteria
    int niter = 0;	 // Counter for iteration
    int niter_Out = 0;	 // Counter for outer iteration

    int gradRank = G.get_shapeY().x(); // dimension of gradient vector at each site
    //cout<<"gradRank="<<gradRank<<endl;

    double res; // Relative residual of data 
    double nTV;				// TV norm of X
    double nL1;				// L1 norm of X

    double nY = Y.matrix().norm();	// norm of Y
    double vobj = 0;			// Value of objective function

    // Initialization
    AX = A.forward(X);  
    GX = G.forward(X);
    X0 = X; X1 = X;
    bool converged = false;

    while (niter < maxIter and not converged) {    
      // Given new X, solve U by shrinkage
      // Solve : min gamma*mu_TV/beta*\sum W_k*|U_k| + 1/2*|GX - Nu/beta - U|^2
      if (aniso)
	U = l1_shrink(GX - Nu/beta, gamma/beta, W);
      else
	U = l2_shrike(GX - Nu/beta, gamma/beta, gradRank, W);

      // Given new X, solve V by shrinkage
      // Solve : min (1-gamma)*mu_L1/alpha*\sum |V_k| + 1/2*|X-Omega/alpha-V|^2
      V = l1_shrink(X - Omega /alpha, (1-gamma)/alpha);

      // Given new U and V, compute X by one-step descent method
      // Update temporary variables first (Lambda, Nu modified at outer iteration)      
      if (eq)
	AXmYmL = AX-Y-Lambda/mu;
      else
	AXmYmL = AX-Y;
      // AXmYmL = (eq) ? (AX-Y-Lambda) : (AX-Y); // This doesnt compile
      GXmUmN = GX-U-Nu/beta;

      // Gradient wrt X of AL function
      //gradX = mu * A.backward(AXmYmL) + beta * G.backward(GXmUmN);
      A.backward(AXmYmL, totoX1);
      G.backward(GXmUmN, totoX2);
      gradX = mu*totoX1 + beta * totoX2 + alpha *(X - V) - Omega;

      if (niter == 0) { // Do steepest descent at 1st iteration
	// Gradient of AL function      
	// AgradX = A.forward(gradX);
	// GgradX = G.forward(gradX);      
	A.forward(gradX, AgradX);
	G.forward(gradX, GgradX);      
	gradstep = (gradX*gradX).sum() / (mu*(AgradX*AgradX).sum() + beta*(GgradX*GgradX).sum() + alpha * (gradX*gradX).sum());
      }
      else {
	// Set gradstep through BB formula
	dgradX = gradX - gradX1;
	gradstep = (dX * dX).sum() / (dX * dgradX).sum();
      }
      
      X -= gradstep * gradX;
      // if (nonneg) // Projection 
      // 	X = (X<0).select(0, X);      
      A.forward(X, AX);
      G.forward(X, GX);
      dX = X - X1;

      // Update inner states of X and U
      RdX_Inn = dX.matrix().norm() / X1.matrix().norm();
      X1 = X;
      gradX1 = gradX;

      if (RdX_Inn <= tol_Inn) {
	Nu -= beta * (GX - U);
	Omega -= alpha * (X - V);
	if (eq)  // Equality constraint
	  Lambda -= mu * (AX - Y);	

	// Save convergence information
	RdX_Out = (niter_Out == 0) ? 1. : (X - X0).matrix().norm() / X0.matrix().norm();
	converged = converged or (RdX_Out <= tol_Out);

	// Update outer states of X and U
	X0 = X;
    
	// Print convergence information
	if (verbose) {
	  res = (AX - Y).matrix().norm();      
	  nTV = GTVNorm(GX, gradRank).sum();
	  nL1 = X.abs().sum();
	  vobj = 0.5 * mu*res*res + gamma*nTV + (1-gamma)*nL1;
	  printf("Iteration %d : \ttol_Out = %1.5e\tvobj = %1.5e\tres = %1.5e\trres = %1.5e\tTV norm = %1.5e\trTV norm = %1.5e\tL1 norm = %1.5e\trL1 norm = %1.5e\n", niter, RdX_Out, vobj, res, res/nY, nTV, nTV/G.get_shapeY().y(),  nL1, nL1/X.size());
	  //printf("               \t\n");

	  // if (verbose > 1 and BI != 0 and imsave != 0 and outpath != 0 and dimObj != 0) {	      
	  //   string fname = *outpath + "/iter_" + Tools::itoa_fixlen3(niter);	      
	  //   ArrayXXd imr = BI->blob2pixel(X, *dimObj);
	  //   (*imsave)(imr, fname, false);
	  // }
	}
	niter_Out++;
      }
      niter++;
    }

    if (niter >= maxIter and verbose)
      cout<<"TVL1AL3 terminated without convergence."<<endl;
    
    res = (AX - Y).matrix().norm();
    nTV = GTVNorm(GX, gradRank).sum();
    nL1 = X.abs().sum();
    vobj = 0.5 * mu*res*res + gamma*nTV + (1-gamma) * nL1;
    return ConvMsg(niter, vobj, res, gamma*nTV + (1-gamma) * nL1);
  }
}
