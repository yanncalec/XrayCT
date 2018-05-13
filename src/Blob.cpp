#include "Blob.hpp"

// Member functions of Class Blob
ArrayXd Blob::Abel(const ArrayXd &r) const
{
  ArrayXd v(r.size());
  for (int n=0; n<r.size(); n++)
    v[n] = this->Abel(r[n]);
  return v;
}

ArrayXd Blob::AbelTable(size_t N) const
{ 
  // generate a strip table of length N, for x \in [-radius, radius)
  //return this->Abel(ArrayXd::LinSpaced(-1 * this->radius, this->radius, N));
  ArrayXd T = ArrayXd::Zero(N);
  for (size_t n=0; n<N; n++)
    T[n] = this->Abel(-radius + n * (2*radius) / N);

  return T;
}

ArrayXd Blob::StripTable(size_t N) const
{ 
  // generate a strip table of length N, for x \in [-1, 1)
  //ArrayXd y=this->Abel(ArrayXd::LinSpaced(-1 * this->radius, this->radius, N));
  //return Tools::cumsum(y * 2 * this->radius / N);
  return Tools::cumsum(this->AbelTable(N)) * 2 * this->radius / N;
  //thrust::transform(y.begin(), y.end(), thrust::make_constant_iterator(2 * this->radius / N), y.begin(), thrust::multiplies<double>());
}

string Blob::type2str(BlobType blobtype)
{
  switch(blobtype) {
  case _GS_ : 
    return "Gaussian";
  case _DIFFGS_ :
    return "Diff-Gaussian";
  case _MEXHAT_ :
    return "Mexican Hat";
  case _D4GS_ :
    return "Fourth order derivative of Gaussian";
  default : 
    return "Unknown";
  }
}

// ofstream & operator<<(ofstream &out, const Blob &B)
// {
//   out.write((char *)&B.radius, sizeof(double));
//   out.write((char *)&B.blobtype, sizeof(BlobType));
//   //out.write((char *)B.parms.data(), sizeof(double) * BLOBPARMLEN);
//   out.write((char *)&B.alpha, sizeof(double));
//   out.write((char *)&B.beta, sizeof(double));
//   out.write((char *)&B.mul, sizeof(double));
//   out.write((char *)&B.dil, sizeof(double));
//   return out;
// }

void Blob::save(ofstream &out) const
{
  out.write((char *)&this->radius, sizeof(double));
  out.write((char *)&this->blobtype, sizeof(BlobType));
  //out.write((char *)this->parms.data(), sizeof(double) * BLOBPARMLEN);
  out.write((char *)&this->alpha, sizeof(double));
  out.write((char *)&this->beta, sizeof(double));
  out.write((char *)&this->mul, sizeof(double));
  out.write((char *)&this->dil, sizeof(double));
}

// template<class T>
// T & operator<<(T &out, const Blob &B) 
// {
//   out<<"----Blob----"<<endl;
//   out<<"Name : "<<B.blobname<<endl;
//   out<<"Radius : "<<B.radius<<endl;
//   out<<"Parameters : "<<B.alpha<<", "<<B.beta<<endl;
//   // for(int n=0; n<B.parms.size(); n++)
//   //   out<<B.parms[n]<<", ";
//   //  out<<endl;
//   out<<"Multiplication : "<<B.mul<<endl;
//   out<<"Dilation : "<<B.dil<<endl;
//   out<<endl;
//   return out;
// }

// template ostream & operator<< <ostream>(ostream &out, const Blob &B);
// template ofstream & operator<< <ofstream>(ofstream &out, const Blob &B);

ostream & operator<<(ostream &out, const Blob &B) 
{
  out<<"----Blob----"<<endl;
  out<<"Name : "<<B.blobname<<endl;
  out<<"Radius : "<<B.radius<<endl;
  out<<"Parameters : "<<B.alpha<<", "<<B.beta<<endl;
  out<<"Multiplication : "<<B.mul<<endl;
  out<<"Dilation : "<<B.dil<<endl;
  return out;
}

ofstream & operator<<(ofstream &out, const Blob &B) 
{
  out<<"----Blob----"<<endl;
  out<<"Name : "<<B.blobname<<endl;
  out<<"Radius : "<<B.radius<<endl;
  out<<"Parameters : "<<B.alpha<<", "<<B.beta<<endl;
  out<<"Multiplication : "<<B.mul<<endl;
  out<<"Dilation : "<<B.dil<<endl;
  return out;
}


// Member functions for class GaussBlob
// double GaussBlob::std_profile(double r) const { 
//   return exp(-this->alpha * r*r);
// }

//! Abel transform of Gaussian blob
double GaussBlob::Abel(double r) const {
  return exp(-this->alpha * r * r) * sqrt(M_PI / this->alpha); 
}  

//! Evaluate the radius of a Gaussian blob with a cut-off error
double GaussBlob::eval_radius(double a, double cut_off_err) {
  return sqrt(-log(cut_off_err) / a); // radius of \Phi
}

//! Evaluate the radius of Fourier transform of a Gaussian blob with a cut-off error
double GaussBlob::eval_Radius(double a, double cut_off_err) {
  return sqrt(-log(cut_off_err) * a) / M_PI; // Radius of \hat\Phi : exp(-PI^2/alpha * R^2)\leq error
}

//! Compute the parameter \f$\alpha\f$ of Gaussian blob corresponding to a FWHM 
double GaussBlob::FWHM2alpha(double f) {
  return pow(M_PI * f, 2.) / log(2);
}

//! Compute the FWHM of a gaussian blob of parameter \f$\alpha\f$ 
double GaussBlob::alpha2FWHM(double a) {
  return sqrt(log(2) * a) / M_PI;
}

// Member functions for class DiffGaussBlob
double DiffGaussBlob::Abel(double r) const { 
  double res = 0;
  for (int n=0; n<this->nbApprx; n++)
    res += this->An(n) / sqrt(3*n + 0.5) * exp(-4*this->alpha / (3*n + 0.5) *r *r);
  
  return 2 * sqrt(M_PI / alpha) * res;
}

//! Using numerical method to evaluate the truncation radius of DiffGaussBlob in space domain. (it's not the frequency truncation radius)
/*!  What is estimated here is the radius \f$r\f$ for a DiffGauss blob
  of parameter $\alpha=1$.  The radius for parameter $\alpha$ is \f$
  r/ \sqrt \alpha \f$.  */
double DiffGaussBlob::eval_radius(double a, double cut_off_err)
{
  double r = 0;
  if(cut_off_err >= 1e-2) 
    r = 0.586510263929618;
  else if (cut_off_err>=1e-3 and cut_off_err < 1e-2) 
    r = 4.1642228739002931;
  else if (cut_off_err>=1e-4 and cut_off_err < 1e-3) 
    r = 8.7976539589442808;
  else
    r = 8.7976539589442808 * 2;
  return r/sqrt(a);
}

//! Evaluate the radius in frequency domain. Remark that \f$ \hat\Psi \f$ by definition is controlled by \f$ \exp(-\pi^2/(8\alpha) \|w\|^2) \f$ .
double DiffGaussBlob::eval_Radius(double a, double cut_off_err)
{
  return sqrt(-8 * log(cut_off_err) * a) / M_PI;
}


// Member functions for class MexHatBlob
double MexHatBlob::Abel(double r) const {
// Abel transform evaluated at r.
  //return -4 * this->alpha * r * r * sqrt(M_PI * this->alpha) * exp(-this->alpha * r * r);
  return (0.5 - this->alpha * r * r) * sqrt(M_PI / this->alpha) * exp(-this->alpha * r * r);
}  

//! Evaluate the radius of the Mexican hat blob

/*!
  let \f$ f(t) = (4(at)^2-2a)\exp(-at^2), a>0 \f$, we want to find a 
  radius \f$ r\geq 1/\sqrt{2a} \f$, such that \f$ |f(t)| \leq \epsilon \f$ for
  \f$ |t| > r \f$. For this, we control the function by a gaussian :
  \f$ |f(t)| \leq c \exp(-d t^2) \f$.
 */
double MexHatBlob::eval_radius(double a, double cut_off_err) {
  // assert(d<this->alpha);
  // double c = pow(2*this->alpha, 2.) * this->beta / (this->alpha-d) * exp(0.5 * d / this->alpha - 0.5);
  // double R = sqrt(-log(cut_off_err / c) / d);
  // return fmax(R, sqrt(1./2/this->alpha));
  double d = 0.9 * a;
  double c = fmax(a/(a-d), 1) * exp((d-a) * 2. / a);
  double R = sqrt(-log(cut_off_err / c) / d);
  return fmax(R, sqrt(2/a));
}

//! Evaluate the radius of the Fourier transform of Mexican hat blob
double MexHatBlob::eval_Radius(double a, double cut_off_err) {
  double mma = M_PI*M_PI/a;
  double d = 0.9 * mma;
  double c = fmax(1/mma, 1/(mma - d)) * mma * mma / M_PI * exp((d-mma) / mma);
  double R = sqrt(-log(cut_off_err / c) / d);
  return fmax(R, sqrt(a)/M_PI);  
}

//!< evaluate the dilation factor in multi-scale representation
double MexHatBlob::eval_dilation(double tol) {
  return 1.5;
}


// Member functions for class D4GaussBlob
double D4GaussBlob::Abel(double r) const {
  return (0.75 + -3 * alpha * r * r + pow(this->alpha * r * r, 2.)) * sqrt(M_PI / this->alpha) * exp(-this->alpha * r * r);
}  

//! Evaluate the radius of the D4Gaussian blob
double D4GaussBlob::eval_radius(double a, double cut_off_err) 
{
  //assert(d<this->alpha);
  double d = 0.9 * a;
  double vv = 1. + sqrt(3.);
  double c = fmax(fmax(2*vv, 4*a*vv/(a-d)),
		  4*a*a/pow(a-d, 2.)) * exp((d-a) * (3+sqrt(3.)) /a);

  double R = sqrt(-log(cut_off_err / c) / d);
  return fmax(R, sqrt((3 + sqrt(3.)) / a));
}

double D4GaussBlob::eval_Radius(double a, double cut_off_err) {
  double mma = M_PI*M_PI/a;
  double d = 0.9 * mma;
  double c0 = pow(M_PI, 5) / pow(a, 3);
  double c = fmax(pow(2/mma, 2), fmax(4/mma/(mma-d), 2/pow(mma-d, 2.))) * c0 * exp((d-mma) * 2 / mma);
  double R = sqrt(-log(cut_off_err / c) / d);
  return fmax(R, sqrt(2*a)/M_PI);  
}

//!< evaluate the dilation factor in multi-scale representation
double D4GaussBlob::eval_dilation(double tol) {
  return 1.5;
}

