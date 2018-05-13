#include "SpAlgo.hpp"
// TVADM implementation
// Nonnegativity is achieved by simple truncation.

namespace SpAlgo {
  ConvMsg TVADM(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X,
		double delta, bool aniso,
		double mu, double beta, // quadratic penealization, always > 0
		double tol_cg, int maxIter_cg,
		double tol, int maxIter, int verbose)
  {
    // A : system operator
    // G : gradient operator
    // Y : data in vector form
    // X : initialization in vector form, output is rewritten on X
    // delta : noise level constraint |Ax-b|<=delta, <0 for normal tv model, ==0 for tveq, >0 for tvdn
    // aniso : true for anisotrope (l1) TV norm
    // mu : data fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // beta : gradient fidelity penalty, less important than mu, could be taken in same order as mu
    // tol_cg : cg iteration tolerance
    // maxIter_cg : maximum iteration steps for CG
    // tol : outer iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    bool dn = (delta > 0);		// true for constraint TVDN model : min |x|_tv st |Ax-b|_2 <= delta
    bool eq = (delta == 0); 		// true for eq TV model: min |x|_tv st Ax=b
    bool tv = (delta < 0);		// true for TV model: min mu/2|Ax-b|^2 + |x|_tv

    if (verbose) {
      cout<<"-----Total Variation minimization by ADM Method-----"<<endl;
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
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      //cout<<"Tikhonov regularization : "<<regId<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    ArrayXd X0, dX;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd U;			// Auxiliary variable GX = U
    ArrayXd Z;			// Auxiliary variable Z = AX-Y, active only for DN model
    ArrayXd Nu, Lambda;		// Lagrangian
    Lambda.setZero(A.get_dimY());
    Nu.setZero(G.get_dimY());

    ArrayXd totoX1(X.size()), totoX2(X.size()), toto;
    const ArrayXd AtY = A.backward(Y);
    const double nY = Y.matrix().norm();	// norm of Y
    const int gradRank = G.get_shapeY().x(); // dimension of gradient vector at each site

    double RdX = 1.; // Relative change in X, inner iteration stopping criteria
    int niter = 0;	 // Counter for iteration
    double vobj = 0;			// Value of objective function
    double res; // Relative residual of data 
    double nTV;				// L0, L1 and TV norm of X

    // Initialization
    ArrayXd AX = A.forward(X);  
    ArrayXd GX = G.forward(X);
    X0 = X; //X1 = X;

    AtAOp AtA(&A);
    AtAOp GtG(&G);
    IdentOp Id(X.size());
    PlusOp *L = new PlusOp(&AtA, &GtG, mu, beta);
    // if (regId==0)
    //   L = new PlusOp(&AtA, &GtG, mu, beta);
    // else {
    //   L = new PlusOp(new PlusOp(&AtA, &GtG, mu, beta), &Id, 1, regId);
    // }

    ConvMsg msg;

    while (niter < maxIter and RdX > tol) {    
      // Given new X, solve Z by projection, for DN model only
      if (dn) {
	toto = AX - Y - Lambda / mu;
	Z = delta * toto / toto.matrix().norm();
      }

      // Given new X, solve U by shrinkage
      // Solve min 1/beta*\sum_k W_k*|U_k| + 1/2*|GX-Nu/beta-U|^2
      if (aniso)
	U = l1_shrink(GX - Nu/beta, 1./beta);
      else
	U = l2_shrike(GX - Nu/beta, 1./beta, gradRank);

      // Given U, compute X by inverting the linear system
      X.setZero();
      G.backward(beta*U+Nu, totoX2);
      if (dn)
	A.backward(mu*(Y+Z)+Lambda, totoX1);
      else if (eq)
	A.backward(mu*Y+Lambda, totoX1);
      else
	totoX1 = mu*AtY;
      // cout<<"mu*AtAX.norm = "<<mu*A.backward(AX).matrix().norm()<<endl;
      // cout<<"beta*DtDX.norm = "<<mu*G.backward(GX).matrix().norm()<<endl;
      msg = LinSolver::Solver_CG(*L, totoX1+totoX2, X, maxIter_cg, tol_cg, false);
	
      // Update multipliers
      A.forward(X, AX);      
      G.forward(X, GX);
      Nu -= beta * (GX - U);
      if (dn)
	Lambda -= mu * (AX - Z - Y);	
      else if (eq)
	Lambda -= mu * (AX - Y);	

      dX = X - X0;

      // Update inner states of X and U
      RdX = (niter == 0) ? 1. : dX.matrix().norm() / X0.matrix().norm();
      X0 = X;
    
      // Print convergence information
      if (verbose) {	
	res = (AX - Y).matrix().norm();      
	nTV = GTVNorm(GX, gradRank).sum();
	vobj = 0.5 * mu*res*res + nTV;
	if(dn)
	  printf("Iteration : %d\tRdX = %1.5e\tvobj = %1.5e\t|AX-Y| = %1.5e\tTV norm = %1.5e\tCG res. = %1.5e\tCG niter. = %d\t|GX-U| = %1.5e\t|AX-Y-Z| = %1.5e\t\n", niter, RdX, vobj, res, nTV, msg.res, msg.niter, (GX-U).matrix().norm(), (AX-Y-Z).matrix().norm());
	else
	  printf("Iteration : %d\tRdX = %1.5e\tvobj = %1.5e\t|AX-Y| = %1.5e\tTV norm = %1.5e\tCG res. = %1.5e\tCG niter. = %d\t|GX-U| = %1.5e\n", niter, RdX, vobj, res, nTV, msg.res, msg.niter, (GX-U).matrix().norm());
      }
      niter++;
    }

    if (niter >= maxIter and verbose)
      cout<<"TVADM terminated without convergence."<<endl;
    
    res = (AX - Y).matrix().norm();
    nTV = GTVNorm(GX, gradRank).sum();
    vobj = 0.5 * mu*res*res + nTV;
    return ConvMsg(niter, vobj, res, nTV);
  }
}
