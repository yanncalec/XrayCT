/*  -*- C -*-  */
/* Blob profile device funcions */

#ifndef _BLOBFUNC_H_
#define _BLOBFUNC_H_

/* Profile function of blob */
/* Gaussian blob */
__device__ inline double std_GS_profile(double r, double alpha)
{
  return exp(-alpha * r * r);
}

/* Diff-Gaussian blob */
__device__ inline double std_DIFFGS_profile(double r, double alpha)
{
  double res = 0;
  double An;
  for (int n=0; n<N_DIFFGS; n++) {
    An = (n==0) ? 1 : exp(log_factorial(2*n) - n * log(4.) - 2*log_factorial(n)) / (1-2.*n);
    res += An / (3*n + 0.5) * exp(-4*alpha / (3*n + 0.5) * r * r);
  }
  return 4 * sqrt(alpha / M_PI) * res;
}

/* Mexcian hat blob */
__device__ inline double std_MEXHAT_profile(double r, double alpha)
{
  //return -(pow(2*alpha*r, 2) - 2 * alpha) * exp(-alpha * r * r);
  return (1 - alpha * r * r) * exp(-alpha * r * r);
}

/* D4-Gaussian (4-th order derivative of Gaussian) blob */
__device__ inline double std_D4GS_profile(double r, double alpha)
{
  return (pow(r*r*alpha, 2) - 4*alpha*r*r + 2) * exp(-alpha * r * r);
}


/* Gradient of blob in x-axis (dX), and y-axis (dY) */
__device__ inline double std_GS_dX(double x, double y, double alpha)
{
  return -2 * alpha * x * exp(-alpha * (x*x + y*y));
}

__device__ inline double std_GS_dY(double x, double y, double alpha)
{
  return -2 * alpha * y * exp(-alpha * (x*x + y*y));
}

__device__ inline double std_DIFFGS_dX(double x, double y, double alpha)
{
  double res = 0;
  double An;
  for (int n=0; n<N_DIFFGS; n++) {
    An = (n==0) ? 1 : exp(log_factorial(2*n) - n * log(4.) - 2*log_factorial(n)) / (1-2.*n);
    res += An / pow(3*n + 0.5, 2.) * exp(-4*alpha / (3*n + 0.5) * (x*x + y*y));
  }
  return -32 * sqrt(alpha / M_PI) * alpha * x * res ;
}

__device__ inline double std_DIFFGS_dY(double x, double y, double alpha)
{
  double res = 0;
  double An;
  for (int n=0; n<N_DIFFGS; n++) {
    An = (n==0) ? 1 : exp(log_factorial(2*n) - n * log(4.) - 2*log_factorial(n)) / (1-2.*n);
    res += An * exp(-4*alpha / (3*n + 0.5) * (x*x + y*y)) / pow(3*n + 0.5, 2.);
  }
  return -32 * sqrt(alpha / M_PI) * alpha * y * res ;
}

__device__ inline double std_MEXHAT_dX(double x, double y, double alpha)
{
  //return -(12*alpha*alpha - 8*alpha*alpha*alpha*(x*x + y*y)) * x * exp(-alpha * (x*x + y*y));
  return (2*alpha*alpha*(x*x + y*y) - 4*alpha) * x * exp(-alpha * (x*x + y*y));
}

__device__ inline double std_MEXHAT_dY(double x, double y, double alpha)
{
  return (2*alpha*alpha*(x*x + y*y) - 4*alpha) * y * exp(-alpha * (x*x + y*y));
}

__device__ inline double std_D4GS_dX(double x, double y, double alpha)
{
  return (-12*alpha + 12*alpha*alpha*(x*x + y*y) - 2*pow((x*x + y*y), 2.)*pow(alpha, 3.)) * x * exp(-alpha * (x*x + y*y));
}

__device__ inline double std_D4GS_dY(double x, double y, double alpha)
{
  return (-12*alpha + 12*alpha*alpha*(x*x + y*y) - 2*pow((x*x + y*y), 2.)*pow(alpha, 3.)) * y * exp(-alpha * (x*x + y*y));
}


/* Abel transform of blob */
__device__ inline double std_GS_Abel(double r, double alpha) {
  return exp(-alpha * r * r) * sqrt(M_PI / alpha); 
}  

__device__ inline double std_DIFFGS_Abel(double r, double alpha) {
  double res = 0;
  double An;
  for (int n=0; n<N_DIFFGS; n++) {
    An = (n==0) ? 1 : exp(log_factorial(2*n) - n * log(4.) - 2*log_factorial(n)) / (1-2.*n);
    res += An / sqrt(3*n + 0.5) * exp(-4*alpha / (3*n + 0.5) * r * r);
  }

  return 2 * sqrt(M_PI / alpha) * res;
}

__device__ inline double std_MEXHAT_Abel(double r, double alpha) {
  return (0.5 - alpha * r * r) * sqrt(M_PI / alpha) * exp(-alpha * r * r);
}  

__device__ inline double std_D4GS_Abel(double r, double alpha) {
  return (0.75 - 3*alpha * r * r + pow(alpha*r*r, 2.)) * sqrt(M_PI / alpha) * exp(-alpha * r * r);
}  

#endif
