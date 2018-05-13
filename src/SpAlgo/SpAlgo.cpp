// Sparse reconstruction algorithms

#include "SpAlgo.hpp"

namespace SpAlgo{

  // Utility functions for TV minimization
    
  ArrayXd GTVNorm(const ArrayXd &X, int gradRank)
  {
    // Calculate the GTV norm of a gradient vector X.
    // The gradient vector is the result of a (generalized) gradient operator,
    // it has dimension K X d, with K the number of sites where
    // the gradient is evaluated, and d is the dimension of a subvector at each
    // site. Typically, for a horizontal and vertical derivative, d=2.
    // The GTV norm is defined as \sum_k |X_k|, where X_k is a \R^d vector, and
    // |.| is the eucledian norm. 

    // Input :
    // X : a flattened gradient vector of dimension K X d, it must be
    // the concatenation of X^1...X^d where X^j\in\R^K is the j-th derivative for
    // all sites.
    // gradRank : the dimension of subvector at each site

    Map<ArrayXXd_ColMajor> Y(X.data(), X.size()/gradRank, gradRank); // By default the mapping use Fortran (column) order
    return (Y*Y).rowwise().sum().sqrt();
  }

  ArrayXd TV_normalize(ArrayXd &X, int gradRank)
  {
    // Normalize a Gradient array inplace and return its TV norm
    int N = X.size()/gradRank;
    ArrayXd Y = ArrayXd::Zero(N);

    for(int n=0; n<N; n++) {
      double v=0;
      for(int d=0; d<gradRank; d++)
	v += pow(X[d*N + n],2);
      Y[n] = sqrt(v);
      if (Y[n] > 0) {
	for(int d=0; d<gradRank; d++)
	  X[d*N+n] /= Y[n];
      }
    }      
    return Y;
  }

  // Orthogonal projection onto l2 ball
  ArrayXd l2_ball_projection(const ArrayXd &X, double radius)
  {
    double rX = X.matrix().norm();
    if (rX <= radius)
      return X;
    else return X/rX * radius;
  }

  // Orthogonal projection onto linf ball
  ArrayXd linf_ball_projection(const ArrayXd &X, double radius)
  {
    radius = fabs(radius);
    ArrayXd Xp = (X < -radius).select(-radius, X);
    return (Xp > radius).select(radius, Xp);
  }

  // l1 reweighted shrinkage operator.
  // Solve min_u  mu * \sum_i w_i|u_i| + 1/2 * ||u - v||^2, where ||.|| is the l2 norm
  // and |.| is the abs value.
  // u_i, v_i, w_i are simply scalars, for i=1..N, and |u-v|^2 = \sum_i (u_i - v_i)^2
  // w_i >= 0
  // The solution is given by shrinkage :
  // u_i = max(|v_i| - w_i*mu, 0) * v_i / |v_i|, where |.| is the abs value

  ArrayXd l1_shrink(const ArrayXd &V, double mu)
  {
    ArrayXd Y = V.abs() - mu;
    return (Y < 0).select(0, Y) * Tools::sign(V);
  }

  ArrayXd l1_shrink(const ArrayXd &V, double mu, const ArrayXd &W)
  {
    ArrayXd Y = V.abs() - W*mu;
    return (Y < 0).select(0, Y) * Tools::sign(V);
  }

  // l2 reweighted shrinkage-like operator.
  // Solve min_u  mu * \sum_i w_i|u_i| + 1/2 * |u - v|^2, where |.| is the l2 norm
  // u_i, v_i are vectors of dimension gradRank, for i=1..N, and |u-v|^2 = \sum_i |u_i - v_i|^2
  // w_i is positive scalar.
  // The solution is given by shrinkage :
  // u_i = max(|v_i| - w_i*mu, 0) * v_i / |v_i|, where |.| is the l2 norm

  ArrayXd l2_shrike(const ArrayXd &V, double mu, int gradRank) 
  {
    ArrayXd Z = V;		// make a copy of memory
    // Convert to matrix form, with each row the gradient like vector at a site
    // Remark that by default Eigen use column major order.
    Map<ArrayXXd_ColMajor> Y(Z.data(), Z.size()/gradRank, gradRank); 
  
    // Normalize Y and return the gradient norm of each row
    // Rowwise means that the operation following rowwise() is done per row
    // Example : A = [[1, 2], [3, 4], [5, 6]], then A.rowwise().sum() = [3, 7, 11], A.colwise().sum() = [9, 12]
    ArrayXd nY = (Y*Y).rowwise().sum().sqrt(); 

    for(int n=0; n<Y.rows(); n++){
      if (nY[n] > mu)
	Y.row(n) *= (nY[n] - mu) / nY[n];
      else
	Y.row(n).setZero();      
    }
    return Z;
  }

  ArrayXd l2_shrike(const ArrayXd &V, double mu, int gradRank, const ArrayXd &W) 
  {
    ArrayXd Z = V;		// make a copy of memory
    // Convert to matrix form, with each row the gradient like vector at a site
    // Remark that by default Eigen use column major order.
    Map<ArrayXXd_ColMajor> Y(Z.data(), Z.size()/gradRank, gradRank); 
  
    // Normalize Y and return the gradient norm of each row
    // Rowwise means that the operation following rowwise() is done per row
    // Example : A = [[1, 2], [3, 4], [5, 6]], then A.rowwise().sum() = [3, 7, 11], A.colwise().sum() = [9, 12]
    ArrayXd nY = (Y*Y).rowwise().sum().sqrt(); 

    for(int n=0; n<Y.rows(); n++){
      if (nY[n] > W[n]*mu)
	Y.row(n) *= (nY[n] - W[n]*mu) / nY[n];
      else
	Y.row(n).setZero();
    }   
    return Z;
  }

  ArrayXd NApprx(const ArrayXd &X, int nterm)
  {
    // const gsl_rng_type * Trng;
    // gsl_rng *rng;     
    // gsl_rng_env_setup();     
    // Trng = gsl_rng_default;
    // rng = gsl_rng_alloc (Trng);
#ifdef _DEBUG_
    assert(nterm > 0 and nterm < X.size());
#endif
    ArrayXd Xabs = X.abs();
    double val = Tools::Nth_maxCoeff(Xabs.data(), X.size(), nterm);
    return (Xabs<=val).select(0, X);
  }

  // void NApprx(ArrayXd &X, int nterm, const gsl_rng *rng)
  // {
  //   if (nterm > 0 and nterm < X.size()) {
  //     ArrayXd Xabs = X.abs();
  //     double val = Tools::Nth_maxCoeff(Xabs.data(), X.size(), nterm, rng);
  //     X = (Xabs<val).select(0, X);
  //   }
  // }

  ArrayXi NApprx_support(const ArrayXd &X, int nterm)
  {   
#ifdef _DEBUG_
    assert(nterm > 0 and nterm < X.size());
#endif
    ArrayXd Xabs = X.abs();
    double val = Tools::Nth_maxCoeff(Xabs.data(), X.size(), nterm);
    //ArrayXi S = (Xabs>val).select(1, ArrayXi::Zeros(X.size()));
    ArrayXi S(X.size());
    for(int n=0; n<S.size(); n++)
      S[n] = (Xabs[n]>val) ? 1 : 0;
    return S;
  }

  ArrayXd NApprx_support_double(const ArrayXd &X, int nterm)
  {   
#ifdef _DEBUG_
    assert(nterm > 0 and nterm < X.size());
#endif
    ArrayXd Xabs = X.abs();
    double val = Tools::Nth_maxCoeff(Xabs.data(), X.size(), nterm);
    //ArrayXi S = (Xabs>val).select(1, ArrayXi::Zeros(X.size()));
    ArrayXd S(X.size());
    for(int n=0; n<S.size(); n++)
      S[n] = (Xabs[n]>val) ? 1 : 0;
    return S;
  }

}

    
  // ArrayXd TVe_X(const ArrayXd &GX, int gradRank, double epsilon)
  // {
  //   // Calculate the l2 norm (relaxed by epsilon) per site of gradient vector GX
  //   Map<ArrayXXd_ColMajor> toto (GX.data(), GX.size()/gradRank, gradRank);
  //   return sqrt((toto * toto).colwise().sum() + epsilon*epsilon);
  // }

  // ArrayXd TVe_X_normalize(ArrayXd &GX, int gradRank, double epsilon)
  // {
  //   // Calculate the l2 norm (relaxed by epsilon) per site of gradient vector GX
  //   // Normalize it and return the norm of each site in vector nGX_e
  //   Map<ArrayXXd_ColMajor> toto (GX.data(), GX.size()/gradRank, gradRank); 
  //   // Eigen use row major by default, GX is the concatenation of two partial derivative vectors.
  //   ArrayXd nGX_e = sqrt((toto * toto).rowwise().sum() + epsilon*epsilon);
  //   //cout<<nGX_e.size()<<", "<<toto.rows()<<endl;
  //   for(int n=0; n<toto.rows(); n++)
  //     toto.row(n) /= nGX_e[n];  
  //   return nGX_e;
  // }

