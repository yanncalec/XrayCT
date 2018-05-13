#include "SpAlgo.hpp"
// TVAL3 BB implementation
// Nonnegativity is achieved by simple truncation.

namespace SpAlgo {
  ConvMsg TVAL3(LinOp &A, LinOp &G, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X, 
		//		ArrayXd &Nu, ArrayXd &Lambda,
		double delta, bool aniso, bool nonneg, double mu, double beta,
		double tol_Inn, double tol_Out, double tol_gap, int maxIter, int verbose)
  {
    // A : system operator
    // G : gradient like operator
    // Y : data in vector form
    // W : weight for TV norm
    // X : initialization in vector form, output is rewritten on X
    // Nu : initial value of lagrange multiplier for GX = U
    // Lambda : initial value of lagrange multiplier for AX = Y constraint
    // delta : noise level constraint |Ax-b|<=delta, <0 for normal tv model, ==0 for tveq, >0 for tvdn
    // aniso : true for anisotrope (l1) TV norm
    // mu : data fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // beta : gradient fidelity penalty, less important than mu, could be taken in same order as mu
    // nonneg : true for positivity constraint
    // tol_Inn : inner iteration tolerance
    // tol_Out : outer iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    bool dn = (delta > 0);		// true for constraint TVDN model : min |x|_tv st |Ax-b|_2 <= delta
    bool eq = (delta == 0); 		// true for eq TV model: min |x|_tv st Ax=b
    bool tv = (delta < 0);		// true for TV model: min mu/2|Ax-b|^2 + |x|_tv

    if (verbose) {
      cout<<"-----Total Variation minimization by AL3 Method-----"<<endl;
      if(tv)
	cout<<"Solve min mu/2 |Ax-b|^2 + |x|_TV"<<endl;
      else if(eq)
	cout<<"Solve min |x|_TV st Ax=b"<<endl;
      else 
	cout<<"Solve min |x|_TV st |Ax-b|<=delta"<<endl;

      cout<<"Parameters :"<<endl;
      cout<<"Data fidelity penalty mu: "<<mu<<endl;
      cout<<"Gradient fidelity penalty beta: "<<beta<<endl;
      cout<<"Noise level delta: "<<delta<<endl;
      cout<<"Positive constaint : "<<nonneg<<endl;
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Inner stopping tolerance : "<<tol_Inn<<endl;
      cout<<"Outer stopping tolerance : "<<tol_Out<<endl;
    }

    ArrayXd X0, X1, dX;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd U, Z;			// Auxiliary variable GX = U, AX-Y=Z
    ArrayXd Nu, Lambda;		// Lagrangian
    Lambda.setZero(A.get_dimY());
    Nu.setZero(G.get_dimY());
  
    ArrayXd gradX, gradX1, dgradX;	// Gradient of AL function wrt X
    ArrayXd AgradX(A.get_dimY()), GgradX(G.get_dimY());	
    ArrayXd AX, GX;		

    ArrayXd totoX1(X.size()), totoX2(X.size()), toto;

    double gradstep;		// gradient step length
    ArrayXd AXmYmL, GXmUmN;

    double RdX_Out = 1.; // Relative change in X, outer iteration stopping criteria
    double RdX_Inn = 1.; // Relative change in X, inner iteration stopping criteria
    int niter = 0;	 // Counter for iteration
    int niter_Out = 0;	 // Counter for outer iteration

    const double nY = Y.matrix().norm();	// norm of Y
    const int gradRank = G.get_shapeY().x(); // dimension of gradient vector at each site
    //cout<<"gradRank="<<gradRank<<endl;

    double res; // Relative residual of data 
    double nTV;				// L0, L1 and TV norm of X
    double vobj = 0;			// Value of objective function

    // Initialization
    AX = A.forward(X);  
    GX = G.forward(X);
    X0 = X; X1 = X;

    bool converged = false;
    //const double tol_gap=2.5e-2;
    double gapA=1, gapG=1;

    //    while (niter < maxIter and RdX_Out > tol_Out) {    
    while (!converged) {    
      // Given new X, solve Z by projection, for DN model only
      if (dn) {
	toto = AX - Y - Lambda / mu;
	Z = l2_ball_projection(toto, delta);
	//Z = delta * toto / toto.matrix().norm();
      }

      // Given new X, solve U by shrinkage
      // Solve min 1/beta*\sum_k W_k*|U_k| + 1/2*|GX-Nu/beta-U|^2
      if (aniso)
	U = l1_shrink(GX - Nu/beta, 1./beta, W);
      else
	U = l2_shrike(GX - Nu/beta, 1./beta, gradRank, W);

      // Given U, Z, compute X by one-step descent method
      // Update temporary variables first (Lambda, Nu modified at outer iteration)
      if (dn)
	AXmYmL = AX-Y-Z-Lambda/mu;
      else if (eq)
	AXmYmL = AX-Y-Lambda/mu;
      else
	AXmYmL = AX-Y;
      GXmUmN = GX-U-Nu/beta;

      // Gradient wrt X of AL function
      //gradX = mu * A.backward(AXmYmL) + beta * G.backward(GXmUmN);
      A.backward(AXmYmL, totoX1);
      G.backward(GXmUmN, totoX2);
      gradX = mu * totoX1 + beta * totoX2;

      if (niter == 0) { // Do steepest descent at 1st iteration
	// Gradient of AL function      
	A.forward(gradX, AgradX);
	G.forward(gradX, GgradX);      
	gradstep = (gradX*gradX).sum() / (mu*(AgradX*AgradX).sum() + beta*(GgradX*GgradX).sum());
      }
      else {
	// Set gradstep through BB formula
	dgradX = gradX - gradX1;
	gradstep = (dX * dX).sum() / (dX * dgradX).sum();
      }
      
      X -= gradstep * gradX;

      if (nonneg) // Projection 
	X = (X<0).select(0, X);      

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
	if (dn)
	  Lambda -= mu * (AX - Z - Y);	
	else if (eq)
	  Lambda -= mu * (AX - Y);	

	// Save convergence information
	RdX_Out = (niter_Out == 0) ? 1. : (X - X0).matrix().norm() / X0.matrix().norm();

	// Update outer states of X and U
	X0 = X;
	gapG = (GX-U).matrix().norm();      
	if (dn)
	  gapA = (AX-Y-Z).matrix().norm();

	// Print convergence information
	if (verbose) {	
	  res = (AX - Y).matrix().norm();      
	  nTV = GTVNorm(GX, gradRank).sum();
	  vobj = 0.5 * mu*res*res + nTV;
	  if (dn)
	    printf("Iteration : %d\tRdX_Out = %1.5e\tvobj = %1.5e\t|AX-Y| = %1.5e\tTV norm = %1.5e\t|GX-U| = %1.5e\t|AX-Y-Z| = %1.5e\t\n", niter, RdX_Out, vobj, res, nTV, gapG, gapA);
	  else
	    printf("Iteration : %d\tRdX_Out = %1.5e\tvobj = %1.5e\t|AX-Y| = %1.5e\tTV norm = %1.5e\t|GX-U| = %1.5e\n", niter, RdX_Out, vobj, res, nTV, gapG);
	}
	niter_Out++;
	if (niter >= maxIter) break;
	if (dn){
	  if (RdX_Out <= tol_Out and (AX-Y).matrix().norm() <= delta and gapG <= tol_gap and gapA <= tol_gap) 
	    converged = true;
	}
	else if (eq) {
	  if (RdX_Out <= tol_Out and (AX-Y).matrix().norm() <= tol_Out and gapG <= tol_gap) 
	    converged = true;
	}
	else
	  if (RdX_Out <= tol_Out and gapG <= tol_gap)
	    converged = true;
      }
      niter++;
    }

    if (!converged and verbose)
      cout<<"TVAL3 terminated without convergence."<<endl;
    
    res = (AX - Y).matrix().norm();
    nTV = GTVNorm(GX, gradRank).sum();
    vobj = 0.5 * mu*res*res + nTV;
    return ConvMsg(niter, vobj, res, nTV);
  }
}
