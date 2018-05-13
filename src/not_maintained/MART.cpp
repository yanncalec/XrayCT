#include "Algo.hpp"
#include "LinSolver.hpp"

namespace Algo{
  ConvMsg MART(LinOp &A, MARTOp &M, const ArrayXd &Y,
	       ArrayXd &X, double tau,
	       double tol, int maxIter, int verbose)
  {
    // Solve the maximization entropy problem : 

    // By MART (Multiplication ART) method
    // The unique solution exists.
    
    if (verbose) {
      cout<<"-----Entropy maximization by MART-----"<<endl;
      cout<<"Solve min sum x_i*log(x_i) st Ax=b"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"MART factor : "<<tau<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    //cout<<X.size()<<endl;
    ArrayXd X0 = X;
    ArrayXd AX(Y.size());
    ArrayXd toto(Y.size());
    double RdX = 1., res, entropy;
    int niter = 0;
    bool decrease = true;
    double res0 = INFINITY;

    while(RdX > tol and niter < maxIter and decrease) {
      //while( niter < maxIter) {
      A.forward(X, AX);

      for (int n=0; n<Y.size(); n++) {
	if (Y[n]<ERR)
	  toto[n] = INFINITY;
	else	
	  toto[n] = tau*(log(Y[n]) -log(AX[n]));
      }

      M.backward(toto.data(), X.data());

      RdX = (X - X0).matrix().norm();
      X0 = X;

      if (verbose and niter % 25 == 0) {	
	//if (verbose) {	
	entropy = 0;
	for(int n=0; n<X.size(); n++)
	    entropy += (X[n]>0) ? X[n] * log(X[n]) : 0;

	res = (AX - Y).matrix().norm();      
	decrease = (res<res0);
	res0 = res;
	printf("Iteration : %d\tRdX = %1.5e\tres = %1.5e\tentropy = %1.5e\n", niter, RdX, res, entropy);
	// printf("X: %f, %f, %f\n", X.minCoeff(), X.maxCoeff(), X.mean());
	// printf("toto: %f, %f, %f\n", toto.minCoeff(), toto.maxCoeff(), toto.mean());
	// printf("AX: %f, %f, %f\n", AX.minCoeff(), AX.maxCoeff(), AX.mean());
      }
      niter++;
    }
   
    if (niter >= maxIter and verbose)
      cout<<"MART terminated without convergence."<<endl;
    
    entropy = 0;
    for(int n=0; n<X.size(); n++) 
      entropy += (X[n]>0) ? X[n] * log(X[n]) : 0;

    res = (AX - Y).matrix().norm();      

    return ConvMsg(niter, entropy, res, 0);
  }
}
