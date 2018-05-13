#include "SpAlgo.hpp"
// L1AL3 BB implementation
// Nonnegativity is achieved by simple truncation.

namespace SpAlgo {
  ConvMsg L1AL3(const LinOp &A, const ArrayXd &W, const ArrayXd &Y,
		ArrayXd &X, ArrayXd &Nu, ArrayXd &Lambda,
		bool eq, bool nonneg, double mu, double beta,
		double tol_Inn, double tol_Out, int maxIter, int verbose)
  {
    // A : system operator
    // Y : data in vector form
    // W : weight for L1 norm
    // X : initialization in vector form, output is rewritten on X
    // Nu : initial value of lagrange multiplier for GX = U
    // Lambda : initial value of lagrange multiplier for AX = Y constraint
    // eq : true for L1 equality model, false for L1/L2 model
    // mu : data fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // beta : gradient fidelity penalty, less important than mu, could be taken in same order as mu
    // nonneg : true for positivity constraint
    // tol_Inn : inner iteration tolerance
    // tol_Out : outer iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    if (verbose) {
      cout<<"-----L1 minimization by AL3 Method-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Positive constaint : "<<nonneg<<endl;
      cout<<"Data fidelity penalty : "<<mu<<endl;
      cout<<"L1 fidelty penalty : "<<beta<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Inner stopping tolerance : "<<tol_Inn<<endl;
      cout<<"Outer stopping tolerance : "<<tol_Out<<endl;
      cout<<"L1 with equality constraint :"<<eq<<endl;
    }

    ArrayXd X0, X1, dX;			// Unknown, X1 for inner state and X0 for outer state
    ArrayXd U;				// Auxiliary variable X = U
  
    ArrayXd gradX, gradX1, dgradX;	// Gradient of AL function wrt X
    ArrayXd AgradX;
    ArrayXd AX;		

    ArrayXd totoX1;
    totoX1.setZero(X.size());

    double gradstep;
    ArrayXd AXmYmL, XmUmN;

    double RdX_Out = 1.; // Relative change in X, outer iteration stopping criteria
    double RdX_Inn = 1.; // Relative change in X, inner iteration stopping criteria
    int niter = 0;	 // Counter for iteration

    double vobj;		   // value of objective function
    double res; // Relative residual of data 
    double nL1;				// L1 norm of X
    double nY = Y.matrix().norm();	// norm of Y

    // Initialization
    AX = A.forward(X);  
    X0 = X; X1 = X;
        
    while (niter < maxIter and RdX_Out > tol_Out) {    
      // Given new X, solve U by shrinkage
      // Solve min mu/beta*\sum_k W_k*|U_k| + 1/2*|X-Nu/beta-U|^2
      U = l1_shrink(X - Nu/beta, mu/beta, &W);

      // Given U, compute X by one-step descent method
      // Update temporary variables first (Lambda, Nu modified at outer iteration)
      if (eq)
	AXmYmL = AX-Y-Lambda/mu;
      else
	AXmYmL = AX-Y;
      XmUmN = X-U-Nu/beta;

      // Gradient wrt X of AL function
      A.backward(AXmYmL, totoX1);
      gradX = totoX1 + beta * XmUmN;

      if (niter == 0) { // Do steepest descent at 1st iteration
	// Gradient of AL function      
	AgradX = A.forward(gradX);
	gradstep = (gradX*gradX).sum() / ((AgradX*AgradX).sum() + beta*(gradX*gradX).sum());
      }
      else {
	// Set alpha through BB formula
	dgradX = gradX - gradX1;
	gradstep = (dX * dX).sum() / (dX * dgradX).sum();
      }
      
      X -= gradstep * gradX;
      if (nonneg) // Projection 
	X = (X<0).select(0, X);      
      AX = A.forward(X);
      dX = X - X1;

      // Update inner states of X and U
      RdX_Inn = dX.matrix().norm() / X1.matrix().norm();
      X1 = X;
      gradX1 = gradX;

      // Update multipliers and enter in next outer iteration when
      // rel.changement of inner iteration is small
      if (RdX_Inn <= tol_Inn) {
	Nu -= beta * (X - U);
	if (eq)  // Equality constraint
	  Lambda -= mu * (AX - Y);	

	// Save convergence information
	RdX_Out = (X - X0).matrix().norm() / X0.matrix().norm();
	// Update outer states of X and U
	X0 = X;
    
	// Print convergence information
	if (verbose) {	
	  nL1 = X.abs().sum();
	  res = (AX-Y).matrix().norm();
	  vobj = 0.5 * res * res + mu * nL1;
	  
	  printf("Iteration : %d\ttol_out = %1.5e\tvobj = %1.5e\tres = %1.5e\trres = %1.5e\tL1 norm = %1.5e\trL1 norm = %1.5e\n", niter, RdX_Out, vobj, res, res/nY, nL1, nL1/X.size());
	}
      }
      niter++;
    }
    if (niter >= maxIter and verbose)
      cout<<"L1AL3 terminates without convergence."<<endl;      

    nL1 = X.abs().sum();
    res = (AX-Y).matrix().norm();
    vobj = 0.5 * res * res + mu * nL1;
    return ConvMsg(niter, vobj, res, nL1);
  }
}
