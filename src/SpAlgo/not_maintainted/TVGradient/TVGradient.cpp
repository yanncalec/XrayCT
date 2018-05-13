#include "SpAlgo.hpp"

namespace SpAlgo {
  ConvMsg TVGradient(const LinOp &A, const LinOp &G, const ArrayXd &Y, 
		     ArrayXd &X, double epsilon, double mu, bool nonneg, double tol, 
		     size_t maxIter, int verbose, const ArrayXd *ref)
  {
    // TV is relaxed to TV_e = \sum_i \sqrt(\norm(GX_i)^2 + \epsilon^2)
    // Its therefore differentiable : DTV_e(x) = G^*(Diag(1/ \sqrt(\norm(GX_i)^2 + \epsilon^2)) * GX)

    // A : system operator
    // G : gradient like operator
    // Y : data in vector form
    // X : initialization in vector form, output is rewritten on X
    // epsilon : relaxation parameter for TV norm
    // mu : data fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // nonneg : true for positivity constraint
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message
    // ref : reference image

    if (verbose) {
      cout<<"-----Total Variation minimization by Gradient Descent Method-----"<<endl;
      cout<<"Parameters : "<<endl;
      cout<<"TV relaxation : "<<epsilon<<endl;
      cout<<"Data fidelity penalty : "<<mu<<endl;
      cout<<"Positive constaint : "<<nonneg<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
      cout<<"Max.  iterations : "<<maxIter<<endl;
    }

    ArrayXd X0;			// Unknown
    ArrayXd AX, GX, GX0;
    ArrayXd nGX_e;

    ArrayXd gradX;	// Gradient of AL function wrt X
    ArrayXd AgradX, GgradX;	
    size_t gradRank = G.shapeY[1]; // Dimension of gradient vector
    size_t nbSites = G.dimY / gradRank; // Be careful, nbSites is not G.dimX

    double alpha;
    ArrayXd H, Z, AH, GH;

    // Initialization

    X0 = X;
    AX = A.forward(X0);  
    GX = G.forward(X0);
       
    double RdX = 1.; // Relative change in X, outer iteration stopping criteria
    size_t niter = 0;	 // Counter for outer iteration

    ConvMsg msg(maxIter);
    //  cout<<"OK 0"<<endl;

    // Outer Iteration
    while (niter++ < maxIter and RdX > tol) {
      // Gradient wrt X of objective function
      GX0 = GX;
      nGX_e = TVTools::TVe_X_normalize(GX0, gradRank, epsilon);
      //    cout<<"OK 1"<<endl;

      gradX = mu*A.backward(AX-Y) + G.backward(GX0) * 2;
      double ngradX = (gradX *gradX).sum();
      AgradX = A.forward(gradX);
      GgradX = G.forward(gradX);      

      // Steepest descent
      ArrayXd toto1 = GgradX*GgradX;
      ArrayXd toto2 = GX*GX*GgradX*GgradX;

      Map<ArrayXXd> V1(toto1.data(), nbSites, gradRank);    
      Map<ArrayXXd> V2(toto2.data(), nbSites, gradRank);
      for(size_t n=0; n<nbSites; n++) {
	V1.row(n) /= nGX_e[n];
	V2.row(n) /= pow(nGX_e[n], 3);
      }
      alpha = ngradX / (mu * (AgradX * AgradX).sum() + V1.sum()-V2.sum());

      //    cout<<"OK 3"<<endl;
    
      Z = X - alpha * gradX;
      if (nonneg) { // Projection gradient method
	Z = (Z<0).select(0, Z);
      }

      H = Z - X;	// Decreasing vector      
      AH = A.forward(H);
      GH = G.forward(H);

      // Backtrack line search
      ArrayXd AXmY = AX-Y;

      double tau = 1.;
      size_t citer = 0;

      double val = (AXmY+tau*AH).matrix().squaredNorm()*mu/2 + TVTools::TVe_X(GX+tau*GH, gradRank, epsilon).sum();
      double val0 = (AXmY).matrix().squaredNorm()*mu/2 + nGX_e.sum();
      double cst = (H * gradX).sum();

      //    cout<<"OK 4"<<endl;
      while (val > val0 + 1e-3 * tau * cst and citer < 10) {
	tau *= 0.5;
	val = (AXmY+tau*AH).matrix().squaredNorm()*mu/2 + TVTools::TVe_X(GX+tau*GH, gradRank, epsilon).sum();	  
	citer++;
      }
      // cout<<"OK 5"<<endl;

      X += tau * H;
      if (citer >= 10) 
	cout<<"Backtrack line search failed."<<endl;

      AX = A.forward(X);
      GX = G.forward(X);     

      RdX = (X - X0).matrix().norm() / X0.matrix().norm();
      msg.RelErr.push_back(RdX);
      if (ref != 0) {
	msg.SNR.push_back(Tools::SNR(X, *ref));
	msg.Corr.push_back(Tools::Corr(X, *ref));
      }

      X0 = X;
      // cout<<"OK 6"<<endl;
         
      // Print convergence information
      if (verbose > 1) {	
	  nTV = TVTools::GTVNorm(GX, gradRank) / G.shapeY[0];
	  res = (AX-Y).matrix().norm() / nY;
      
	  if (ref == 0)
	    printf("Iteration : %lu\ttol_Out = %2.5f\tres. = %2.5f\tTV norm = %f\n", niter, msg.RelErr.back(), res, nTV);
	  else
	    printf("Iteration : %lu\ttol_Out = %2.5f\tCorr = %2.5f\tSNR = %3.5f\tres. = %2.5f\tTV norm = %f\n", niter, msg.RelErr.back(), msg.Corr.back(), msg.SNR.back(), res, nTV);
      }
    }
    return msg;
  }
}
