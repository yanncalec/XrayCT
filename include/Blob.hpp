#ifndef _BLOB_H_
#define _BLOB_H_

#include "Types.hpp"
#include "Tools.hpp"

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

//! Abstract class for standard blob basis
/*!
  The convention of blob function is : \f$ \mu \Phi(\delta x;p_1, p_2,...)\f$, with \f$\Phi(x; p_1, p_2...) \f$ the standard blob function, depending on
  parameters \f$ p_1, p_2... \f$.  \f$ \mu, \delta \f$ are the multiplication and dilation constants, useful in multi-scale blob system.
 */
class Blob {
protected :
  //ArrayXd parms;

public :    
  const double radius;		//!< radius of blob (truncation)
  const string blobname;	//!< name of blob function
  const BlobType blobtype;	//!< Profile type of blob
  const double alpha;		//!< Parameter set for blob function.
  const double beta;		
  const double mul;		//!< Multiplication constant, it has effect only on GPU kernel calls
  const double dil;		//!< Dilation constant, it has effect only on GPU kernel calls

  Blob(double radius, BlobType ptype, double mul, double dil, double alpha, double beta = 0) 
    : radius(radius), blobname(Blob::type2str(ptype)), blobtype(ptype), alpha(alpha), beta(beta), mul(mul), dil(dil) {}

  Blob(const Blob &B) : radius(B.radius), blobname(B.blobname), blobtype(B.blobtype), alpha(alpha), beta(beta), mul(B.mul), dil(B.dil) {}
  
  virtual double Abel(double) const = 0;    //!< Abel transform of blob : \f$\int_{R} b_\Phi(\sqrt{\|y\|^2 + t^2}) dt \f$

  ArrayXd Abel(const ArrayXd &) const; //!< Vectorized Abel transform : \f$ \mu/\delta \mathcal{A}\Phi(\|\delta x\|) \f$
  ArrayXd AbelTable(size_t) const; //!< Direct footprint table
  ArrayXd StripTable(size_t) const; //!< Strip integration table

  static string type2str(BlobType ptype);
    
  void save(ofstream &) const; 	//!< Save a Blob object to output file stream

  //template<class T> friend T & operator<<(T &out, const Blob &B);
  friend ostream & operator<<(ostream &, const Blob &);
  friend ofstream & operator<<(ofstream &, const Blob &);
};


//! Standard Gaussian blob.
/*!
  Expression of Gaussian blob \f$ \Phi(x) \f$ :  
  \f$ \Phi(x) = \exp(-\alpha \|x\|^2) \f$, and
  \f$\Phi(x)=0 \f$ if \f$\|x\|\f$ is larger than blob radius.
  - parms[0] =\f$ \alpha \f$
*/
class GaussBlob : public Blob { 

public :    
  static double eval_radius(double a, double cut_off_err=1e-4);
  static double eval_Radius(double a, double err=1e-4);
  static double FWHM2alpha(double f);
  static double alpha2FWHM(double a);

  GaussBlob(double r, double a, double mul = 1., double dil = 1.)
    : Blob(r, _GS_, mul, dil, a){}

  double Abel(double) const;
};


//! Difference Gaussian (DiffGauss) blob.
/*! 
  DiffGauss blob \f$ \Psi_0 \f$ is defined in frequency domain as :
  \f$ |\hat \Psi_0(w)|^2 = \hat\Phi(w/2) - \hat\Phi(w) \f$, with \f$\Phi(x) = \exp(-\alpha\|x\|^2) \f$
  the Gauss blob. In space it has the asymptotic expression :
  \f$ \Psi_0(x) = 4\sqrt{\alpha/\pi} \sum_{k\geq 0} A_k/(3k+1/2) \exp(-4\alpha / (3k+1/2) \|x\|^2) \f$, with
  the constant \f$ A_k = (2k)!/(4^k(k!)^2(1-2k)) \f$
*/

class DiffGaussBlob : public Blob { 
private :
  //! The value of coefficient \f$ A_n \f$ in the asymptotic expansion
  static double An(int n) { return (n==0) ? 1 : (exp(Tools::log_factorial(2*n) - n * log(4.) - 2 * Tools::log_factorial(n)) / (1-2.*n)); }
  // //! The normalized Gaussian blob \f$ \bar\Phi \f$
  // static double nmlGaussBlob(double r, double a) { return (a/M_PI) * exp(-a * r * r); }

public : 
  const int nbApprx;		//!< Number of asymptotic expansion, parms[1]

  static double eval_radius(double a, double cut_off_err=1e-4);
  static double eval_Radius(double a, double cut_off_err=1e-4);

  DiffGaussBlob(double r, double a, int N = 100, double mul = 1., double dil = 1.)
    : Blob(r, _DIFFGS_, mul, dil, a, N), nbApprx(N) {} 

  double Abel(double) const;

  friend ostream & operator<<(ostream &, const DiffGaussBlob &);  
  //friend ofstream & operator<<(ofstream &, const DiffGaussBlob &);  
};

//! Mexican hat blob
/*!
  Mexican hat is the negative of the second order derivative of Gaussian \f$ \exp(-\alpha\|x\|^2) \f$ :
  \f$ \Psi_0(x) = -(4\alpha^2\|x\|^2 - 2\alpha)\exp(-\alpha\|x\|^2) \f$
*/
class MexHatBlob : public Blob { 
private :

public : 
  static double eval_radius(double a, double cut_off_err=1e-4);
  static double eval_Radius(double a, double cut_off_err=1e-4);
  double eval_dilation(double tol=1e-6);

  MexHatBlob(double r, double a, double mul = 1., double dil = 1.)
    : Blob(r, _MEXHAT_, mul, dil, a) {} 

  double Abel(double) const;

  friend ostream & operator<<(ostream &, const DiffGaussBlob &);  
  //friend ofstream & operator<<(ofstream &, const DiffGaussBlob &);  
};


//! Fourth order derivative Gaussian blob
/*!
  D4Gauss is the 4th order derivative of Gaussian  \f$ \exp(-\alpha\|x\|^2) \f$ :
  \f$ \Psi_0(x) = \sqrt\pi(16\alpha^4\|x\|^4 - 48\alpha^3\|x\|^2+12\alpha^2)\exp(-\alpha\|x\|^2) \f$  
 */

class D4GaussBlob : public Blob {

public : 
  static double eval_radius(double a, double cut_off_err=1e-4);
  static double eval_Radius(double a, double cut_off_err=1e-4);
  double eval_dilation(double tol=1e-6);

  D4GaussBlob(double r, double a, double mul = 1., double dil=1.)
    : Blob(r, _D4GS_, mul, dil, a) {} 

  double Abel(double) const;

  friend ostream & operator<<(ostream &, const DiffGaussBlob &);  
  //friend ofstream & operator<<(ofstream &, const DiffGaussBlob &);  
};

#endif
