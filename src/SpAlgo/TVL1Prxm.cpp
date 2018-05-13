#include "SpAlgo.hpp"
// TV-L1 Proximal method
// This method doesn't introduce auxiliary variable X=V.
// In order to solve the X-step, it linearizes around current solution X^k and add a proximal term
// Therefore the sparsity of X is obtained in a direct way

namespace SpAlgo {
  ConvMsg TVL1Prxm(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X,
		   double delta, double gamma, 
		   double muA, double muG, double muI, bool aniso,
		   double tol, int maxIter, 
		   double stol, double sparsity, int verbose, int Freq,
		   void (*savearray)(const ArrayXd &, const string &), const string * outpath)
  {
    // A : system operator
    // G : gradient like operator
    // Y : data in vector form
    // X : initialization in vector form, output is rewritten on X
    // delta : noise level constraint |Ax-b|<=delta, <0 for normal tv-l1 model, ==0 for eq, >0 for dn
    // gamma : TV to L1 weight
    // muA : data fidelity penalty
    // muG : gradient fidelity penalty
    // muI : identity fidelity penalty (proximal penalty)
    // aniso : true for anisotrope (l1) TV norm
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    bool dn = (delta > 0);		// true for constraint BPDN model : min (1-gamma)*|x|_tv+gamma*|x|_1 st |Ax-b|_2 <= delta
    bool eq = (delta == 0); 		// true for eq BP model: min (1-gamma)*|x|_tv+gamma*|x|_1 st Ax=b
    bool tvl1 = (delta < 0);		// true for l1 model: min 1/2*mu_A*|Ax-b|^2 + (1-gamma)*|x|_tv+gamma*|x|_1

    if (verbose) {
      cout<<"-----Total Variation L1 minimization by ADM Method-----"<<endl;
      if (tvl1)
	cout<<"Solve min mu/2 |Ax-b|^2 + (1-gamma)*|x|_TV + gamma*|x|_1"<<endl;
      else if(eq)
	cout<<"Solve min (1-gamma)*|x|_TV + gamma*|x|_1 st Ax=b"<<endl;
      else 
	cout<<"Solve min (1-gamma)*|x|_TV + gamma*|x|_1 st |Ax-b|<=delta"<<endl;

      cout<<"Parameters :"<<endl;
      cout<<"Anisotrope TV norm : "<<aniso<<endl;
      cout<<"TV to L1 weight : "<<gamma<<endl;
      cout<<"Data fidelity : "<<muA<<endl;
      cout<<"Gradient penalty : "<<muG<<endl;
      cout<<"Identity (proximal) penalty : "<<muI<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    int sFreq = (int)fmax(Freq/5., 1);		// Check the support change every sFreq iterations.
    //int Freq = 50;		// print message every Freq iterations.

    ArrayXd X0, dX, dX0;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd U;			// Auxiliary variable GX = U
    ArrayXd Z;			// Auxiliary variable Z = AX-Y, active only for DN model
    ArrayXd LambdaA, LambdaG, LambdaI;		// Lagrangian
    LambdaA.setZero(A.get_dimY());
    LambdaG.setZero(G.get_dimY());
    LambdaI.setZero(X.size());
  
    ArrayXd gradX(X.size());

    ArrayXd totoX1(X.size()), totoX2(X.size()), totoX3(X.size()), totoY;
    //const ArrayXd AtY = A.backward(Y);
    //const double nY = Y.matrix().norm();	// norm of Y
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

    ArrayXi XSupp;		// Support of X
    ArrayXi XSupp0 = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
    int Xdiff = X.size();
    double rXdiff=1.;		// Relative size of support change

    ArrayXi MSupp;		// Main support (nnz-largest coeffs)
    MSupp.setZero(X.size());	
    ArrayXi MSupp0 = XSupp0;
    int Mdiff = X.size();
    double rMdiff=1.;		// Relative size of main support change
    const int nnz = (int)ceil(X.size() * sparsity);

    double RddX = 1, RdX = 1.; // Relative change in X
    int niter = 0;	 // Counter for iteration
    ConvMsg msg;

    bool converged = false;

    while (niter < maxIter and not converged) {    
      // Given new X, solve Z by shrinkage
      if(dn) {
	// totoY = AX - Y - LambdaA / muA;
	// Z = totoY / totoY.matrix().norm() * delta;
	Z = l2_ball_projection(AX - Y - LambdaA / muA, delta);
      }

      // Given new X, solve U by shrinkage
      if (aniso)
	U = l1_shrink(GX - LambdaG/muG, (1-gamma)/muG);
      else
	U = l2_shrike(GX - LambdaG/muG, (1-gamma)/muG, gradRank);
      
      // Solve new X by proximal + shrinkage
      // Gradient
      if (dn)
	A.backward(AX - Y - Z - LambdaA/muA, totoX1);
      else if (eq)
	A.backward(AX - Y - LambdaA/muA, totoX1);
      else
	A.backward(AX - Y, totoX1);

      G.backward(GX - U - LambdaG/muG, totoX2);
      //cout<<"totoX2.norm="<<totoX2.matrix().norm()<<endl;
      gradX = muA*totoX1 + muG*totoX2;	

      X = l1_shrink(X - gradX / muI, gamma/muI);

      // Update multipliers
      A.forward(X, AX);      
      G.forward(X, GX);
      // if (niter > 1 and niter %20 ==0) 
      // 	{
      if (dn)
	LambdaA -= muA * (AX - Z - Y);	
      else if (eq)
	LambdaA -= muA * (AX - Y);	
      LambdaG -= muG * (GX - U);
      //      }
      dX = X - X0;

      // Update inner states of X and U
      RdX = (niter == 0) ? 1. : dX.matrix().norm() / X0.matrix().norm();
      RddX = (niter < 2) ? 1. : (dX0-dX).matrix().norm() / dX0.matrix().norm();
      X0 = X;
      dX0 = dX;

      if (niter % sFreq == 0) {
	// Support changement of X
	XSupp = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
	Xdiff = (XSupp0 - XSupp).abs().sum();
	XSupp0 = XSupp;
	rXdiff = Xdiff*1./X.size();
	converged = rXdiff<stol;

	if (sparsity > 0 and sparsity < 1) {
	  MSupp = SpAlgo::NApprx_support(X, nnz);
	  Mdiff = (MSupp0 - MSupp).abs().sum();
	  MSupp0 = MSupp;
	  rMdiff = Mdiff*1./X.size();
	  converged = converged or (rMdiff < fmin(stol, 5e-4));
	}
      }

      converged = converged or (RdX < tol and RddX < 1e-1);

      if (niter % Freq == 0 and savearray != NULL and outpath != NULL){
	char buffer[256];
	sprintf(buffer, "%s/tempo_%d", outpath->c_str(), niter);
	savearray(X, buffer);
      }
      
      // Print convergence information
      if (verbose and niter % Freq == 0) {	
	//      if (verbose) {	
	res = (AX - Y).matrix().norm();      
	nTV = GTVNorm(GX, gradRank).sum();
	nL1 = X.abs().sum();
	nL0 = Tools::l0norm(X);
	vobj = 0.5 * muA*res*res + (1-gamma)*nTV + gamma * nL1;

	if(dn)
	  printf("Iteration : %d\tRdX = %1.2e\tRddX = %1.2e\tvobj = %1.2e\t|AX-Y| = %1.2e\tTV norm = %1.2e\t|GX-U| = %1.2e\t|AX-Y-Z| = %1.2e\tL1 norm = %1.2e\tnon zero per. = %1.3e\tXdiff=%d\trXdiff=%1.2e\tMdiff=%d\trMdiff=%1.2e\n", niter, RdX, RddX, vobj, res, nTV, (GX-U).matrix().norm(), (AX-Y-Z).matrix().norm(), nL1, nL0/X.size(), Xdiff, rXdiff, Mdiff, rMdiff);
	else
	  printf("Iteration : %d\tRdX = %1.2e\tRddX = %1.2e\tvobj = %1.2e\t|AX-Y| = %1.2e\tTV norm = %1.2e\t|GX-U| = %1.2e\tL1 norm = %1.2e\tnon zero per. = %1.3e\tXdiff=%d\trXdiff=%1.2e\tMdiff=%d\trMdiff=%1.2e\n", niter, RdX, RddX, vobj, res, nTV, (GX-U).matrix().norm(), nL1, nL0/X.size(), Xdiff, rXdiff, Mdiff, rMdiff);
	
      }
      niter++;
    }

    if (niter >= maxIter and verbose)
      cout<<"TVL1-Proximal terminated without convergence."<<endl;
    
    res = (AX - Y).matrix().norm();
    nTV = GTVNorm(GX, gradRank).sum();
    nL1 = X.abs().sum();
    vobj = 0.5 * muA*res*res + (1-gamma)*nTV + gamma * nL1;
      return ConvMsg(niter, vobj, res, (1-gamma)*nTV + gamma * nL1);
  }
}
