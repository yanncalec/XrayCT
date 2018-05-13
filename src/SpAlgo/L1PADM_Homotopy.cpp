#include "SpAlgo.hpp"
#include "BlobProjector.hpp"
// L1 minimization by Homotopy continuation with PADM

namespace SpAlgo {
  ConvMsg L1PADM_Homotopy(LinOp &A, LinOp &Q, const ArrayXd &W, const ArrayXd &Y, ArrayXd &X,
			  double delta, double beta, double tau, double gamma,
			  double tol, int maxIter, double sparsity, 
			  double stol, int hIter, double incr, int debias, int verbose)
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
      cout<<"-----L1 minimization by PADM method with homotopy continuation of parameter mu-----"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"PADM beta : "<<beta<<endl;
      cout<<"PADM max. iterations : "<<maxIter<<endl;
      cout<<"PADM stopping tolerance : "<<tol<<endl;
      cout<<"PADM support chg. stopping tolerance : "<<stol<<endl;
      cout<<"Iteration number of homotopy continuation : "<<hIter<<endl;
      cout<<"Homotopy incremental factor : "<<incr<<endl;
      cout<<"CG debiasing max. iterations : "<<debias<<endl;
    }

    BlobProjector *P = dynamic_cast<BlobProjector *>(const_cast<LinOp *>(&Q));
    // Used only by general projector
    DiagOp M(X.size(), 1.);
    CompOp *AM = new CompOp(&A, &M);
    AtAOp L(AM);

    ConvMsg* msg = new ConvMsg[hIter];
    double beta0 = beta*pow(incr, hIter-1);
    double sp0 = sparsity * pow(incr, hIter-1);
    //int maxIter0 = int(AP.maxIter/AP.cont);
    //double stol0 = AP.stol/pow(AP.incr, AP.cont-1);
    //double mu = AP.mu*W.abs().sum();
    // cout<<maxIter0<<endl;
    // cout<<mu0<<endl;
    for(int n=0; n<hIter; n++) {
      if (verbose and hIter>1)
	printf("\nHomotopy continuation sub-iteration : %d, beta = %1.5e, sparsity = %1.5e\n",n, beta0, sp0);

      // if (n==hIter-1)		// Special treatement for last iteration
      // 	msg[n] = L1PADM(*P, W, Y, X, beta, tol, maxIter, 0, sparsity, verbose-1);
      // else
      msg[n] = L1PADM(A, W, Y, X, 
		      delta, 0, beta0, tau, gamma,
		      tol, maxIter, stol, sp0, verbose-1);
      beta0 /= incr;
      sp0 /= incr;

      if (verbose) {
	int nL0 = Tools::l0norm(X);
	printf("Niter = %d\t|Ax-b| = %1.5e\tL1 norm = %1.5e\tnon zero per. = %1.5e\n", msg[n].niter, msg[n].res, msg[n].norm, nL0*1./X.size());
      }

      if (debias>0) {
	if (verbose) {
	  cout<<"Debiasing by Conjugate Gradient method..."<<endl;  
	  cout<<"Residual |Ax-b| before debiasing = "<<(A.forward(X)-Y).matrix().norm()<<endl;
	}

	//int nnz = min((int)ceil(X.size() * sp0), Tools::l0norm(X)); // The main support size

	if (P==NULL) {
	  //ArrayXd S = SpAlgo::NApprx_support_double(X, nnz);
	  ArrayXd S = (X.abs()>0).select(1, ArrayXd::Zero(X.size()));
	  M.set_a(S);
	  //LinSolver::Solver_normal_CG(A, Y, X, 100, 1e-2, &MSupp, verbose);
	  LinSolver::Solver_CG(L, AM->backward(Y), X, debias, 1e-3, false);
	}
	else {
	  vector<ArrayXd> XS = P->separate(X);
	  vector<ArrayXb> S; S.resize(XS.size());
	  for (int n=0; n<XS.size(); n++) {
	    S[n] = (XS[n].abs()>0).select(true, ArrayXb::Constant(XS[n].size(), false));
	  }

	  // ArrayXd Sd = SpAlgo::NApprx_support_double(X, nnz);
	  // vector<ArrayXd> S0 = P->separate(Sd);
	  // for (int n=0; n<XS.size(); n++) {
	  //   S[n] = (S0[n]>0).select(true, ArrayXb::Constant(S0[n].size(), false));
	  // }

	  P->set_maskBlob(S);
	  AtAOp L(&A);
	  LinSolver::Solver_CG(L, A.backward(Y), X, debias, 1e-3, false);
	  P->reset_maskBlob();
	}
	
	if (verbose) {	
	  cout<<"Residual |Ax-b| after debiasing = "<<(A.forward(X)-Y).matrix().norm()<<endl;
	}
      }
    }

    double nL1 = X.abs().sum();
    double res = (A.forward(X)-Y).matrix().norm();
    double vobj = nL1;
    int niter;
    for (int n=0; n<hIter; n++)
      niter += msg[n].niter;

    return ConvMsg(niter, vobj, res, nL1);
  }
}



    // // Supplementary iterations to enhance convergence precision -- BEGIN
    // double sincr=0.9;
    // int sIter = (int)ceil(log(1e-5/stol) / log(sincr));
    // double stol0 = stol;

    // for(int n=0; n<sIter; n++) {
    //   if (verbose and hIter>1)
    // 	printf("\nSupplementary iteration : %d of %d, PADM support chg. stopping tolerance : %f\n",n+1, sIter, stol0);
    //   ConvMsg msg = L1PADM(*P, W, Y, X, mu, tol, maxIter, stol0, verbose-1);
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
