// Classic iterative reconstruction algorithms

#include "Algo.hpp"
#include "LinSolver.hpp"
#define ERR 1e-5

namespace Algo{

  ConvMsg Landwebber(LinOp &P, const ArrayXd &B, ArrayXd &X, 
		     bool nonneg, double tol, int maxIter, bool verbose)
  {
    if (verbose) {
      cout<<"-----Landwebber iteration-----"<<endl;
      cout<<"Solve min |A*x-b|^2 s.t. x>=0"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Positivity constraint : "<<nonneg<<endl;      
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    int niter=0;
    ArrayXd grad(P.get_dimX()), grad1, dgrad;
    ArrayXd Ag(P.get_dimY()), Ap(P.get_dimY()), Ax(P.get_dimY());
    ArrayXd X0(X.size()), dX, U, Z, p;

    double gradstep;
    double RdX = 1, res = 0;

    while (RdX > tol and niter < maxIter) 
      {
    	P.forward(X, Ax);	  // Ax = A * x
    	P.backward(Ax - B, grad); // Gradient vector grad = A^t(Ax - B)
    	P.forward(grad, Ag);		   // A * gradient

	if (niter==0)
	  gradstep = grad.matrix().squaredNorm()/Ag.matrix().squaredNorm();  // Steepest descent
	else {
	  // Set gradstep through BB formula
	  dgrad = grad - grad1;
	  gradstep = (dX * dX).sum() / (dX * dgrad).sum();
	}
    	// alpha = fmin(1., fmax(1e-5, alpha));
    	// if (nonneg) {
    	//   U = X - alpha*grad;
    	//   Z = (U<0).select(0, U); //Z[Z<0]=0;
    	//   p = X - Z; // Real descent direction
	//   P.forward(p, Ap);
    	// }
    	// else {
    	//   p = grad;
	//   Ap = Ag;
	// }

    	// beta = fmax(1e-3, fmin(0.999, (p * grad).sum()/Ap.matrix().squaredNorm()));
    	// X = X - beta*p ;

	X -= gradstep*grad;
	if (nonneg)
	  X = (X<0).select(0, X); 

	dX = X-X0;
	RdX = dX.matrix().norm() / X0.matrix().norm();
    	X0 = X;
	grad1 = grad;

    	if (verbose and niter % 25 ==0) {
	  res = (Ax-B).matrix().norm();
	  printf("Iteration : %d, RdX=%1.5e\tres=%1.5e\tgradstep=%1.5e\n", niter, RdX, res, gradstep);
    	}
    	niter++;
      }
    return ConvMsg(niter, res, res, 0);
  }

  double ARTUR_delta(double t, EPPType priortype, double alpha){
    // detla(t)= p'(t)/t  if t>0
    //         = p''(0+)  if t==0
    t = fabs(t);
    switch (priortype){
    case _EPP_GM_:		// p(t)=alpha*t^2/(1+alpha*t^2), p'(t)/t=2*alpha/(1+alpha*t^2)^2 -> 2*alpha
      return (t>0)? 2*alpha/pow(alpha + t*t, 2.) : 2/alpha;
    case _EPP_HS_: //p(t)=sqrt{\alpha+t^2}, p'(t)/t=1/sqrt{alpha+t^2} -> 1/sqrt{alpha}
      return (t>0)? 1/sqrt(alpha+t*t) : 1/sqrt(alpha);
    case _EPP_GR_: //p(t)=log{cosh(alpha*t)}, p'(t)/t=(t==0) : alpha^2 ? alpha*tanh(alpha*t)/t -> alpha^2
      return (t>1e-6)? alpha* tanh(alpha*t)/t : alpha*alpha;
    case _EPP_GS_:		//p(t)=(alpha*t^2)/(1+alpha*t^2), p'(t)/t=2*alpha*exp(-alpha*t^2) -> 2*alpha
      return (t>0)? 2*alpha*exp(-alpha*t*t) : 2*alpha;
    default:
      cerr<<"Unknown type of Edge-Preserving prior!"<<endl;
      exit(0);
    }
    return 0;
  }

  double ARTUR_phi(double t, EPPType priortype, double alpha){
    // detla(t)= p'(t)/t  if t>0
    //         = p''(0+)  if t==0
    t = fabs(t);
    switch (priortype){
      // Convex priors: small alpha is better
    case _EPP_HS_: 
      return sqrt(alpha+t*t);
    case _EPP_GR_: 
      return log(cosh(alpha * t));
      // Non convex priors: large alpha is better
    case _EPP_GM_:
      return t*t/(alpha+t*t);
    case _EPP_GS_:
      return 1-exp(-alpha*t*t);
    default:
      cerr<<"Unknown type of Edge-Preserving prior!"<<endl;
      exit(0);
    }
    return 0;
  }

  ConvMsg ARTUR(LinOp &A, LinOp &G, const ArrayXd &Y,
		ArrayXd &X, EPPType priortype, double alpha, double mu, 
		double cgtol, int cgmaxIter, double tol, int maxIter, int verbose)
  {
    if (verbose) {
      cout<<"-----Half-Qudratic minimization by ARTUR Method-----"<<endl;
      cout<<"Solve min |Ax-b|^2 + mu*sum_j phi(|Gx_j|)"<<endl;
      cout<<"Parameters :"<<endl;
      cout<<"Edge-Preserving Prior type : "<<priortype<<endl;
      cout<<"Edge-Preserving Prior parameter : "<<alpha<<endl;
      cout<<"Regularization weight : "<<mu<<endl;
      cout<<"CG Max. iterations : "<<cgmaxIter<<endl;
      cout<<"CG stopping tolerance : "<<cgtol<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
    }

    AtAOp *AtA = new AtAOp(&A);
    AtDAOp *GtDG = new AtDAOp(&G, ArrayXd::Ones(G.get_dimY()));
    PlusOp *L = new PlusOp(AtA, GtDG, 1, 1);
    //cout<<L.get_dimX()<<" "<<L.get_dimY()<<endl;

    const ArrayXd AtY = A.backward(Y);
    const double nY = Y.matrix().norm();
    double RdX = 1.;
    int niter = 0;
    double res=0, nPhi=0, vobj=0;

    ArrayXd GX; GX.setZero(G.get_dimY());
    ArrayXd B; B.setZero(G.get_dimY());
    ArrayXd AX; AX.setZero(A.get_dimY());
    ArrayXd X0 = X;
    ArrayXd toto;

    while (niter < maxIter and RdX > tol) {
      // Fix X, solve T:
      G.forward(X, GX);
      //printf("GX.abs() : min=%f\tmax=%f\n",GX.abs().minCoeff(), GX.abs().maxCoeff());
      for(int n=0; n<GX.size(); n++) 
	B[n]=ARTUR_delta(GX[n], priortype, alpha);
	//B[n]=2*alpha*exp(-alpha*GX[n]*GX[n]);
      //printf("B : min=%f\tmax=%f\n",B.minCoeff(), B.maxCoeff());

      // Fix T, solve X
      toto = 0.5 * mu * B;
      GtDG->set_a(toto);
      //printf("toto : min=%f\tmax=%f\n",toto.minCoeff(), toto.maxCoeff());
      X.setZero();
      ConvMsg msg = LinSolver::Solver_CG(*L, AtY, X, cgmaxIter, cgtol, false); // Reduce CG precision can result non convergence of HQ algorithm

      RdX = (niter==0) ? 1 : (X-X0).matrix().norm() / X0.matrix().norm();
      X0 = X;

      if (verbose and niter % 5 == 0) {	
	//if (verbose) {	
	A.forward(X, AX);
	G.forward(X, GX);
	nPhi = 0;
	for(int n=0; n<GX.size(); n++) 
	  nPhi += ARTUR_phi(GX[n], priortype, alpha);

	res = (AX - Y).matrix().norm();      
	vobj = res*res + mu*nPhi;

	printf("Iteration : %d\tRdX = %1.5e\tvobj = %1.5e\tres = %1.5e\trres = %1.5e\tnPhi = %1.5e\tCG niter=%d\tCG res = %1.5e\n", niter, RdX, vobj, res, res/nY, nPhi, msg.niter, msg.res);
      }
      niter++;
    }
   
    if (niter >= maxIter and verbose)
      cout<<"ARTUR terminated without convergence."<<endl;
    
    nPhi = 0;
    for(int n=0; n<GX.size(); n++) 
      nPhi += ARTUR_phi(GX[n], priortype, alpha);

    res = (AX - Y).matrix().norm();      
    vobj = res*res + mu*nPhi;

    return ConvMsg(niter, vobj, res, nPhi);
  }

  double BesovPrior(const ArrayXd &X, double p, double s)
  {
    // Return the Besov prior
    // Computational friendly version, see paper: "Bayesian Multiresolution method for local tomo in dental X-ray imaging"
    // \sum_k |c_j0,k|^p + \sum_{j>=j0, k, l} 2^{jps} |c_{j,k,l}|^p

    // Non standard wavelet coeffient index <-> scale relationship
    // in R^d, scale number starts by 0: 
    // coarse scale 0 has 1 coefficient
    // the detail scale J=0,1.. 
    // the total number of coefficients started from coarse scale 0 to detail scale J is:
    // (2^d-1)*\sum_{j=0..J} 2^{jd} + 1 = 2^{(J+1)d}
    // so the detaile scale J has its coefficients index : 2^(Jd).. 2^{(J+1)d} - 1
    // The above holds if the wvl coeffs are stored scale by scale, which is not the case for GSL non standard wvl transform.

    double res=0;
    if (s==0) {
      for(int i=0; i<X.size(); i++)
	res += pow(fabs(X[i]), p); 
    }
    else {
      int n = (int)sqrt(X.size());
      assert(n*n == X.size());

      for(int i=0; i<X.size(); i++) {
	div_t q = div(i, n);
	int idx = max(q.quot, q.rem);
	int scale = (idx>0) ? (int)floor(log2(idx*1.)) : 0;
	// int scale = (i>0) ? (int)floor(log2(i+1.)/dim) : 0;
	res += pow(fabs(X[i]), p) * pow(2, scale*p*s); 
      }
    }
    return res;
  }

  ArrayXd BesovPriorGrad(const ArrayXd &X, double p, double s)
  {
    // Return the gradient of Besov prior 
    ArrayXd Y = ArrayXd::Zero(X.size());

    if (s==0) {
      for(int i=0; i<X.size(); i++)
	Y[i] = p * pow(fabs(X[i]), p-1) * (X[i]>0 ? 1 : -1);
    }
    else {
      int n = (int)sqrt(X.size());
      assert(n*n == X.size());

      for(int i=0; i<X.size(); i++) {
	div_t q = div(i, n);
	int idx = max(q.quot, q.rem); // 2^J <= idx <= 2^(J+1) - 1 with J the detail scale
	int scale = (idx>0) ? (int)floor(log2(idx*1.)) : 0;
	//printf("scale=%d\n", scale);
	Y[i] = p * pow(fabs(X[i]), p-1) * (X[i]>0 ? 1 : -1) * pow(2, scale*p*s);
	//res += pow(fabs(X[n]), p) * pow(pow(2, scale), p*s); 
      }
    }
    return Y;    
  }

  ConvMsg WaveletBesov(LinOp &A, const ArrayXd &Y,
		       ArrayXd &X, double mu, double besov_p, double besov_s,
		       double tol, int maxIter, int verbose)
  {
    if (verbose) {
      cout<<"-----Wavelet-Besov prior minimization by nonlinear CG(Polak-Ribière) method-----"<<endl;
      cout<<"Solve min |A*x-b|^2 + mu{|x|_besov(p, s)}^p"<<endl;

      cout<<"Parameters :"<<endl;
      cout<<"Regularization weight : "<<mu<<endl;
      cout<<"Besov prior p : "<<besov_p<<endl;
      cout<<"Besov prior s(s = a + d*(1/2 - 1/p)) : "<<besov_s<<endl;
      cout<<"Max. iterations : "<<maxIter<<endl;
      cout<<"Stopping tolerance : "<<tol<<endl;
      //cout<<"Polak-Ribiere nonlinear CG method : "<<(cgmethod==0)<<endl;
    }

    // Initialization
    //X.setZero();
    ArrayXd X0 = X;
    ArrayXd AXmY = A.forward(X) - Y, AtAXmY = A.backward(AXmY);
    //ArrayXd gradX = -1.*A.backward(Y), gradX0 = gradX;
    ArrayXd gradX = AtAXmY + mu*BesovPriorGrad(X, besov_p, besov_s);
    ArrayXd gradX0 = gradX;
    double ngradX = gradX.matrix().norm(), ngradX0=ngradX;
    ArrayXd bgradX = ArrayXd::Zero(X.size());
    ArrayXd D = -gradX;
    ArrayXd AD = A.forward(D);
    ArrayXd AX = A.forward(X);
    double RdX = 1.;
    int niter = 0;
    double res=0, nBesov=0, vobj=0;
    double beta; 		// CG direction step length

    do {
      // Determine the step length in direction D by line search
      int citer = 0;		// counter for line search
      double alpha = 1e-3;	// descent step length
      double val0 = AXmY.matrix().squaredNorm()/2 + mu*BesovPrior(X, besov_p, besov_s);
      double val = AXmY.matrix().squaredNorm()/2 + pow(alpha*AD.matrix().norm(),2)/2
	+ alpha*(AXmY * AD).sum() + mu*BesovPrior(X + alpha * D, besov_p, besov_s);
      double slope = (gradX*D).sum();
      double nAXmY = AXmY.matrix().squaredNorm();
      double nAD = AD.matrix().norm();
      double dot1 = (AXmY * AD).sum();

      while (val > val0 + (1e-3)*alpha*slope and citer < 32) { // descent direction is -alpha*gradX
	alpha *= 0.5;
	val = nAXmY/2 + pow(alpha*nAD,2)/2 + alpha*dot1 + mu*BesovPrior(X + alpha * D, besov_p, besov_s);
	//printf("val=%1.5e\tval0=%1.5e\tval0-(1e-3)*alpha*slope=%1.5e\n",val, val0, val0 - (1e-3)*alpha*slope);
	citer++;
      }
      
      // Update X
      X += alpha * D;
      RdX = alpha * D.matrix().norm();
      A.forward(X, AX);
      AXmY = AX - Y;
      A.backward(AXmY, AtAXmY);

      // new gradient \nabla F(x)    
      bgradX = BesovPriorGrad(X, besov_p, besov_s);
      gradX = AtAXmY + mu*bgradX;
      ngradX = gradX.matrix().squaredNorm();
      //gradX = AtAXmY + mu*BesovPriorGrad(X, besov_p);

      // Update conjugate direction D
      bool dfailed = false;	// conjugate direction fail flag      
      //beta = ngradX / ngradX0; // Fletcher-Reeves(slower convergence)
      beta = ((gradX - gradX0) * gradX).sum()/ngradX0; // Polak-Ribière
      D = beta * D - gradX;
      if ((D*gradX).sum() > 0) {	// Verify that D is a descent direction       
	D = -gradX;
	dfailed = true;
      }
      A.forward(D, AD);
      gradX0 = gradX;
      ngradX0 = ngradX;

      if (verbose and niter % 25 == 0) {
	//if (verbose) {
	res = AXmY.matrix().norm();	
	nBesov = BesovPrior(X, besov_p, besov_s);
	if (dfailed)
	  printf("CG iteration %d\tngradX=%1.5e\talpha=%1.5e\tciter=%d\tRdX=%1.5e\tres.=%1.5e\tnBesov=%1.5e, conjugate direction failed\n", niter, sqrt(ngradX), alpha, citer, RdX, res, nBesov);
	else
	  printf("CG iteration %d\tngradX=%1.5e\talpha=%1.5e\tciter=%d\tRdX=%1.5e\tres.=%1.5e\tnBesov=%1.5e\n", niter, sqrt(ngradX), alpha, citer, RdX, res, nBesov);
      }

      niter++;
    } while(niter<maxIter and sqrt(ngradX) > tol and RdX > tol);

    if (niter >= maxIter and verbose)
      cout<<"Nonlinear CG terminated without convergence."<<endl;
    
    res = AXmY.matrix().norm();      
    //res = RdX; // residual is the gradient norm
    nBesov = BesovPrior(X, besov_p, besov_s);
    vobj = AXmY.matrix().squaredNorm()/2 + mu*nBesov;

    return ConvMsg(niter, vobj, res, nBesov); 
  }

}
