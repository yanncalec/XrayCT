#include "SpAlgo.hpp"
// TV-L1ADM 
// This method introduces the auxiliary variable X=V, and make threshold on V
// On X it solves a linear system by CG, so numerically not very efficent.

namespace SpAlgo {
  ConvMsg TVL1ADM(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X, ArrayXd &U, ArrayXd &V,
		  double delta, double gamma, double muA, double muG, double muI, bool aniso,
		  double tol_cg, int maxIter_cg, double tol, int maxIter, int verbose)
  {
    // A : system operator
    // G : gradient like operator
    // Y : data in vector form
    // X : initialization in vector form, output is rewritten on X
    // delta : noise level constraint |Ax-b|<=delta, <0 for normal tv-l1 model, ==0 for eq, >0 for dn
    // gamma : L1 weight
    // muA : data fidelity penalty
    // muG : gradient fidelity penalty
    // muI : identity fidelity penalty
    // aniso : true for anisotrope (l1) TV norm
    // tol_cg : CG iteration tolerance
    // maxIter_cg : CG maximum iteration steps
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    bool dn = (delta > 0);		// true for constraint BPDN model : min |x|_tv+gamma*|x|_1 st |Ax-b|_2 <= delta
    bool eq = (delta == 0); 		// true for eq BP model: min |x|_tv+gamma*|x|_1 st Ax=b
    bool tvl1 = (delta < 0);		// true for l1 model: min 1/2*mu_A*|Ax-b|^2 + |x|_tv+gamma*|x|_1

    if (verbose) {
      cout<<"-----Total Variation L1 minimization by ADM Method-----"<<endl;
      if (tvl1)
	cout<<"Solve min mu/2 |Ax-b|^2 + |x|_TV + gamma*|x|_1"<<endl;
      else if(eq)
	cout<<"Solve min |x|_TV + gamma*|x|_1 st Ax=b"<<endl;
      else 
	cout<<"Solve min |x|_TV + gamma*|x|_1 st |Ax-b|<=delta"<<endl;

      cout<<"Parameters :"<<endl;
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      cout<<"L1 weight : "<<gamma<<endl;
      cout<<"Data fidelity : "<<muA<<endl;
      cout<<"Gradient penalty : "<<muG<<endl;
      cout<<"Identity penalty : "<<muI<<endl;
      cout<<"CG Max. iterations : "<<maxIter_cg<<endl;
      cout<<"CG stopping tolerance : "<<tol_cg<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    ArrayXd X0, dX;			// Unknown, X1 for inner state and X0 for outer state
    // ArrayXd U;			// Auxiliary variable GX = U
    // ArrayXd V;			// Auxiliary variable X = V
    ArrayXd Z;			// Auxiliary variable Z = AX-Y, active only for DN model
    ArrayXd LambdaA, LambdaG, LambdaI;		// Lagrangian
    LambdaA.setZero(A.get_dimY());
    LambdaG.setZero(G.get_dimY());
    LambdaI.setZero(X.size());
  
    // CG system operator
    AtAOp AtA(&A);
    AtAOp GtG(&G);
    IdentOp Id(X.size());
    PlusOp *L = new PlusOp(new PlusOp(&AtA, &GtG, muA, muG), &Id, 1, muI);

    ArrayXd totoX1(X.size()), totoX2(X.size()), totoX3(X.size()), totoY;
    const ArrayXd AtY = A.backward(Y);
    const double nY = Y.matrix().norm();	// norm of Y
    const int gradRank = G.get_shapeY().x(); // dimension of gradient vector at each site

    double vobj = 0;			// Value of objective function
    double res;				// Relative residual of data 
    double nTV;				// TV norm of X
    double nL0;				// L0 norm of X
    double nL0_V;			// L0 norm of V
    double nL1;				// L1 norm of X

    // Initialization
    ArrayXd AX, GX;		
    AX = A.forward(X);  
    GX = G.forward(X);
    X0 = X;

    double RdX = 1.; // Relative change in X
    int niter = 0;	 // Counter for iteration
    ConvMsg msg;

    while (niter < maxIter and RdX > tol) {    
      // Given new X, solve Z by shrinkage
      if(dn) {
	// totoY = AX - Y - LambdaA / muA;
	// Z = totoY / totoY.matrix().norm() * delta;
	Z = l2_ball_projection(AX - Y - LambdaA / muA, delta);
      }

      // Given new X, solve U by shrinkage
      if (aniso)
	U = l1_shrink(GX - LambdaG/muG, 1/muG);
      else
	U = l2_shrike(GX - LambdaG/muG, 1/muG, gradRank);

      // Given new X, solve V by shrinkage
      V = l1_shrink(X - LambdaI /muI, gamma/muI);

      // Given new B, U and V, compute X by CG
      if (dn)
	A.backward(muA*(Y+Z) + LambdaA, totoX1);
      else if (eq)
	A.backward(muA*Y + LambdaA, totoX1);
      else
	totoX1 = muA*AtY;
      G.backward(muG*U + LambdaG, totoX2);
      totoX3 = muI*V + LambdaI;

      X.setZero();
      msg = LinSolver::Solver_CG(*L, totoX1+totoX2+totoX3, X, maxIter_cg, tol_cg, false);
      
      // Update multipliers
      A.forward(X, AX);      
      G.forward(X, GX);
      if (dn)
	LambdaA -= muA * (AX - Z - Y);	
      else if (eq)
	LambdaA -= muA * (AX - Y);	
      LambdaG -= muG * (GX - U);
      LambdaI -= muI * (X - V);

      dX = X - X0;

      // Update inner states of X and U
      RdX = (niter == 0) ? 1. : dX.matrix().norm() / X0.matrix().norm();
      X0 = X;

      // Print convergence information
      if (verbose) {	
	res = (AX - Y).matrix().norm();      
	nTV = GTVNorm(GX, gradRank).sum();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);
	nL0_V = Tools::l0norm(V);
	vobj = 0.5 * muA*res*res + nTV + gamma * nL1;

	if(dn)
	  printf("Iteration : %d\tRdX = %1.2e\tvobj = %1.2e\t|AX-Y| = %1.2e\tTV norm = %1.2e\tCG res. = %1.2e\tCG niter. = %d\t|X-V| = %1.2e\t|GX-U| = %1.2e\t|AX-Y-Z| = %1.2e\tL1 norm = %1.2e\tnon zero per. = %1.2e, %1.2e(V)\n", niter, RdX, vobj, res, nTV, msg.res, msg.niter, (X-V).matrix().norm(), (GX-U).matrix().norm(), (AX-Y-Z).matrix().norm(), nL1, nL0/X.size(), nL0_V/X.size());
	else
	  printf("Iteration : %d\tRdX = %1.2e\tvobj = %1.2e\t|AX-Y| = %1.2e\tTV norm = %1.2e\tCG res. = %1.2e\tCG niter. = %d\t|X-V| = %1.2e\t|GX-U| = %1.2e\tL1 norm = %1.2e\tnon zero per. = %1.2e, %1.2e(V)\n", niter, RdX, vobj, res, nTV, msg.res, msg.niter, (X-V).matrix().norm(), (GX-U).matrix().norm(), nL1, nL0/X.size(), nL0_V/X.size());
      }
      niter++;
    }

    if (niter >= maxIter and verbose)
      cout<<"TVL1ADM terminated without convergence."<<endl;
    
    res = (AX - Y).matrix().norm();
    nTV = GTVNorm(GX, gradRank).sum();
    nL1 = X.abs().sum();
    vobj = 0.5 * muA*res*res + nTV + gamma * nL1;
    nL1 = X.abs().sum();
    return ConvMsg(niter, vobj, res, nTV + gamma * nL1);
  }
}
