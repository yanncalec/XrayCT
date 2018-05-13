#include "LinSolver.hpp"
#include "BlobProjector.hpp"

namespace LinSolver{
  ConvMsg Solver_CG(LinOp &A, const ArrayXd &Y, ArrayXd &X, int maxIter, double tol, bool verbose)
  {
    // Solve the linear system Ax=y with A symmetric

#ifdef _DEBUG_
    assert(A.get_dimY() == A.get_dimX());
    assert(X.size() == Y.size());
#endif

    ArrayXd toto(X.size());
    ArrayXd R = Y - A.forward(X); // Redidual

    ArrayXd P = R;		    // Conjugate direction
    ArrayXd AP = ArrayXd::Zero(X.size()); 
    double alpha;

    double nR0 = (R*R).sum();
    double nR1 = nR0;

    int n = 0;
    // cout<<sqrt(nR1)<<", "<<tol<<endl;
    while(n<min(maxIter, (int)X.size()) and sqrt(nR1) > tol) {
      A.forward(P, AP);

      alpha = nR0 / (P * AP).sum();
      X += alpha * P;
      R -= alpha * AP;
      nR1 = (R*R).sum();
      P = R + (nR1 / nR0) * P;
      nR0 = nR1;
      if (verbose and n % 25 == 0) {
	printf("CG iteration %d : res. = %1.5e\n", n, sqrt(nR1));
      }
      n++;
    }
    // if (verbose)
    //   printf("CG iteration %d : res. = %1.5e\n", n, sqrt(nR1));
    //double res = (A.forward(X)-Y).matrix().norm();

    return ConvMsg(n, 0, sqrt(nR1), 0);
  }

  ConvMsg Solver_normal_CG(LinOp &A, const ArrayXd &Y0, ArrayXd &X, int maxIter, double tol, const ArrayXi *Mask0, bool verbose)
  {
    // Solve the linear system Ax=y0 by the normal system : At.Ax=At.y0
    // If Diagonal/Mask preconditionner is used, then sovle : AM x = y by normal system : MtAt. AM x = MtAt y

#ifdef _DEBUG_
    assert(Y0.size() == A.get_dimY());
    assert(X.size() == A.get_dimX());
    if (Mask0 != NULL)
      assert(X.size() == Mask0->size());    
#endif

    //ArrayXd X = ArrayXd::Zero(A.get_dimX()); 
    //X.setZero();
    ArrayXd Mask = (*Mask0 > 0).select(1., ArrayXd::Zero(Mask0->size())); // convert to double
    bool maskon = (Mask0 != NULL);
    ArrayXd totoX(X.size());
    ArrayXd totoY(Y0.size());
    ArrayXd R;			// Residual

    ArrayXd Y;
    if (maskon) {
      Y = A.backward(Y0) * Mask;
      A.forward(X*Mask, totoY);
      A.backward(totoY, totoX);
      R = Y - totoX * Mask;
    }
    else {
      Y = A.backward(Y0);
      R = Y - A.backward(A.forward(X));
    }

    ArrayXd P = R;		    // Conjugate direction
    ArrayXd AP = ArrayXd::Zero(X.size()); 
    double alpha;

    double nR0 = (R*R).sum();
    double nR1 = nR0;

    int n = 0;
    // cout<<sqrt(nR1)<<", "<<tol<<endl;
    while(n<min(maxIter, (int)X.size()) and sqrt(nR1) > tol) {
      // A._forward(P.data(), toto);
      // A._backward(toto, AP.data());
      if (maskon) {
	A.forward(P*Mask, totoY);
	A.backward(totoY, AP);
	AP *= Mask;
      }
      else {
	A.forward(P, totoY);
	A.backward(totoY, AP);
      }

      alpha = nR0 / (P * AP).sum();
      X += alpha * P;
      R -= alpha * AP;
      nR1 = (R*R).sum();
      P = R + (nR1 / nR0) * P;
      nR0 = nR1;
      if (verbose and n % 25 == 0) {
	printf("CG iteration %d : res. = %1.5e\n", n, sqrt(nR1));
      }
      n++;
    }
    // if (verbose)
    //   printf("CG iteration %d : res. = %1.5e\n", n, sqrt(nR1));

    return ConvMsg(n, 0, sqrt(nR1), 0);
  }

  ConvMsg Projector_Debiasing(LinOp &A, const ArrayXd &Y, ArrayXd &X, int maxIter, double tol, bool verbose)
  {
    // Solve the linear system A_s x = y by CG with initialized x and
    // uniquely on its support s (debiasing).

    BlobProjector *P = dynamic_cast<BlobProjector *>(const_cast<LinOp *>(&A));
    // Used only by general projector
    DiagOp M(X.size(), 1.);
    CompOp *AM = new CompOp(&A, &M);
    AtAOp L(AM);
    if (verbose) {
      cout<<"Debiasing by Conjugate Gradient method..."<<endl;  
      cout<<"Residual |Ax-b| before debiasing = "<<(A.forward(X)-Y).matrix().norm()<<endl;
    }

    //int nnz = min((int)ceil(X.size() * sp0), Tools::l0norm(X)); // The main support size
    ConvMsg msg;
    if (P==NULL) {
      //ArrayXd S = SpAlgo::NApprx_support_double(X, nnz);
      ArrayXd S = (X.abs()>0).select(1, ArrayXd::Zero(X.size()));
      M.set_a(S);
      //LinSolver::Solver_normal_CG(A, Y, X, 100, 1e-2, &MSupp, verbose);
      msg = LinSolver::Solver_CG(L, AM->backward(Y), X, maxIter, tol, false);
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
      AtAOp L(P);
      msg = LinSolver::Solver_CG(L, P->backward(Y), X, maxIter, tol, false);
      P->reset_maskBlob();
    }
	
    if (verbose) {	
      cout<<"Residual |Ax-b| after debiasing = "<<(A.forward(X)-Y).matrix().norm()<<endl;
    }
    return msg;
  }

}
