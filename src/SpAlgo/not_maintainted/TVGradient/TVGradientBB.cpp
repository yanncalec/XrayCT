#include "SpAlgo.hpp"

namespace SpAlgo {
  ConvMsg TVGradient(const LinOp &A, const LinOp &G, const ArrayXd &Y, 
		     ArrayXd &X, double epsilon, double mu, bool nonneg, double tol, 
		     size_t maxIter, int verbose, const ArrayXd *ref, 
		     void (*imsave)(const ArrayXXd &, const string &),
		     const BlobImage *BI, const Array2i *dimObj, const string *outpath)
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
    // imsave : function handel for saving image (see CmdTools::imsave)
    // BI : BlobImage object with blob2pixel method, this is useful for saving intermediate results into a gif image
    // dimObj : desired screen image dimension
    // outpath : output path name for intermediate images

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
    ArrayXd nGX_e, dX;   

    ArrayXd gradX, gradX0, dgradX;	// Gradient of AL function wrt X
    ArrayXd AgradX, GgradX;	
    size_t gradRank = G.shapeY[1]; // Dimension of gradient vector

    double alpha, nTV, res;
    double nY = Y.matrix().norm();

    // Initialization
    X0 = X;
    AX = A.forward(X0);  
    GX = G.forward(X0);
       
    double RdX = 1.; // Relative change in X, outer iteration stopping criteria
    size_t niter = 0;	 // Counter for outer iteration

    ConvMsg msg(maxIter);

    // Outer Iteration
    while (niter < maxIter and RdX > tol) {
      // Gradient wrt X of objective function
      GX0 = GX;
      nGX_e = TVTools::TVe_X_normalize(GX0, gradRank, epsilon);
      //    cout<<"OK 1"<<endl;

      gradX = mu*A.backward(AX-Y) + G.backward(GX0) * 2;
      double ngradX = (gradX *gradX).sum();
      AgradX = A.forward(gradX);
      GgradX = G.forward(gradX);      

      if (niter == 0) {
	// Do steepest descent at first iteration
	ArrayXd toto1 = GgradX*GgradX;
	ArrayXd toto2 = GX*GX*GgradX*GgradX;

	Map<ArrayXXd> V1(toto1.data(), G.shapeY[0], gradRank);    
	Map<ArrayXXd> V2(toto2.data(), G.shapeY[0], gradRank);
	for(size_t n=0; n<G.shapeY[0]; n++) {
	  V1.row(n) /= nGX_e[n];
	  V2.row(n) /= pow(nGX_e[n], 3);
	}
	alpha = ngradX / (mu * (AgradX * AgradX).sum() + V1.sum()-V2.sum());
      }
      else {   // BB steplength
	// alpha = (dX * gradX).sum() / ngradX;
	dgradX = gradX - gradX0;
	alpha = (dX * dX).sum() / (dX * dgradX).sum();
      }

      X -= alpha * gradX;
      if (nonneg) { // Projection gradient method
	X = (X<0).select(0, X);
      }

      AX = A.forward(X);
      GX = G.forward(X);     

      dX = X-X0;
      RdX = dX.matrix().norm() / X0.matrix().norm();
      X0 = X;
      gradX0 = gradX;

      msg.RelErr.push_back(RdX);
      if (ref != 0) {
	msg.SNR.push_back(Tools::SNR(X, *ref));
	msg.Corr.push_back(Tools::Corr(X, *ref));
      }

      // Print convergence information
      if (verbose and niter % 10 == 0) {	// Register every 10 image
	nTV = TVTools::GTVNorm(GX, gradRank).sum() / G.shapeY[0];
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
      niter++;
    }
    return msg;
  }
}
