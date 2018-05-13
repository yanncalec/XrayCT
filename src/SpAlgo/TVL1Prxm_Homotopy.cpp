#include "SpAlgo.hpp"
#include "BlobProjector.hpp"

namespace SpAlgo {
  ConvMsg TVL1Prxm_Homotopy(LinOp &A, LinOp &G, const ArrayXd &Y, ArrayXd &X, double gamma, 
			    double muA, double muG, double muI, bool aniso, 
			    double tol, int maxIter, double stol, double sparsity, 
			    int hIter, double incr, int debias, bool noapprx, int verbose, int Freq)
  {
    // A : system operator
    // Y : data in vector form
    // W : weight for L1 norm
    // X : initialization in vector form, output is rewritten on X
    // mu : L1 fidelity penalty, the most important argument, should be set according to noise level, nbProj, pixDet, sizeObj, and nbNode
    // tol : iteration tolerance
    // maxIter : maximum iteration steps
    // verbose : display convergence message

    if (verbose) {
      cout<<"-----TV-L1 minimization by Proximal method with homotopy continuation of parameter mu-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Objective data fidelity penalty : "<<muA<<endl;
      cout<<"TV to L1 weight : "<<gamma<<endl;
      cout<<"TVL1 max. iterations : "<<maxIter<<endl;
      cout<<"TVL1 stopping tolerance : "<<tol<<endl;
      cout<<"TVL1 support chg. stopping tolerance : "<<stol<<endl;
      cout<<"Iteration number of homotopy continuation : "<<hIter<<endl;
      cout<<"Homotopy incremental factor : "<<incr<<endl;
      cout<<"CG debiasing max. iterations : "<<debias<<endl;
    }

    // BlobProjector *P = dynamic_cast<BlobProjector *>(const_cast<LinOp *>(&A));
    // // Used only by general projector
    // DiagOp M(X.size(), 1.);
    // CompOp *AM = new CompOp(&A, &M);
    // AtAOp L(AM);

    ConvMsg* msg = new ConvMsg[hIter];
    double muA0 = muA*pow(incr, hIter-1);
    double sp0 = sparsity * pow(incr, hIter-1);
    const int gradRank = G.get_shapeY().x(); // dimension of gradient vector at each site
    ArrayXd GX;
    double nTV, nL1, res, vobj;

    for(int n=0; n<hIter; n++) {
      if (verbose and hIter>1)
	printf("\nHomotopy continuation sub-iteration : %d, mu = %1.5e, sparsity = %1.5e\n",n, muA0, sp0);

      if (n==hIter-1 and noapprx)
	msg[n] = TVL1Prxm(A, G, Y, X, 
			  -1, gamma, muA, muG, muI, aniso,
			  tol, maxIter, 0, 0, verbose-1, Freq);
      else
	msg[n] = TVL1Prxm(A, G, Y, X, 
			  -1, gamma, muA0, muG, muI, aniso,
			  tol, maxIter, stol, sp0, verbose-1, Freq);

      //msg[n] = L1FISTA(A, W, Y, X, mu0, tol, maxIter, stol, sparsity, verbose-1);

      muA0 /= incr;
      sp0 /= incr;

      if (verbose) {
	int nL0 = Tools::l0norm(X);
	GX = G.forward(X);
	nTV = SpAlgo::GTVNorm(GX, gradRank).sum();
	printf("Niter = %d\t|Ax-b| = %1.5e\tTV norm = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\n", msg[n].niter, msg[n].res, nTV, X.abs().sum(), nL0*1./X.size());
      }

      if ((debias>0 and n<hIter-1) || (debias>0 and n==hIter-1 and !noapprx))
	LinSolver::Projector_Debiasing(A, Y, X, debias, 1e-3, verbose);
    }

    GX = G.forward(X);
    nTV = SpAlgo::GTVNorm(GX, gradRank).sum();
    nL1 = X.abs().sum();
    res = (A.forward(X)-Y).matrix().norm();
    vobj = 0.5 * muA * res * res + (1-gamma) *nTV + gamma* nL1;
    int niter;
    for (int n=0; n<hIter; n++)
      niter += msg[n].niter;

    return ConvMsg(niter, vobj, res, (1-gamma) *nTV + gamma* nL1);
  }
}



    // // Supplementary iterations to enhance convergence precision -- BEGIN
    // double sincr=0.9;
    // int sIter = (int)ceil(log(1e-5/stol) / log(sincr));
    // double stol0 = stol;

    // for(int n=0; n<sIter; n++) {
    //   if (verbose and hIter>1)
    // 	printf("\nSupplementary iteration : %d of %d, FISTA support chg. stopping tolerance : %f\n",n+1, sIter, stol0);
    //   ConvMsg msg = L1FISTA(*P, W, Y, X, mu, tol, maxIter, stol0, verbose-1);
    //   stol0 *= sincr;

    //   if (verbose) {
    // 	int nL0 = Tools::l0norm(X);
    // 	printf("Niter = %d\t|Ax-b| = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\n", msg.niter, msg.res, msg.norm, nL0*1./X.size());
    //   }

    //   if (debias>0) {
    // 	if (verbose) {
    // 	  cout<<"Debiasing by Conjugate Gradient method..."<<endl;  
    // 	  cout<<"Residual |Ax-b| before debiasing = "<<(A.forward(X)-Y).matrix().norm()<<endl;
    // 	}

    // 	if (P==NULL) {
    // 	  ArrayXd S = (X.abs()>0).select(1, ArrayXd::Zero(X.size()));
    // 	  M.set_a(S);
    // 	  //LinSolver::Solver_normal_CG(A, Y, X, 100, 1e-2, &MSupp, verbose);
    // 	  LinSolver::Solver_CG(L, AM->backward(Y), X, debias, 1e-3, false);
    // 	}
    // 	else {
    // 	  vector<ArrayXd> XS = P->separate(X);
    // 	  vector <ArrayXb> S; S.resize(XS.size());
    // 	  for (int n=0; n<XS.size(); n++) {
    // 	    S[n] = (XS[n].abs()>0).select(true, ArrayXb::Constant(XS[n].size(), false));
    // 	  }
    // 	  P->set_maskBlob(S);
    // 	  AtAOp L(P);
    // 	  LinSolver::Solver_CG(L, P->backward(Y), X, debias, 1e-3, false);
    // 	  P->reset_maskBlob();
    // 	}
	
    // 	if (verbose) {	
    // 	  cout<<"Residual |Ax-b| after debiasing = "<<(A.forward(X)-Y).matrix().norm()<<endl;
    // 	}
    //   }
    // }
    // // Supplementary iterations to enhance convergence precision -- END
