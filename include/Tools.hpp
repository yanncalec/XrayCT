#ifndef _TOOLS_H_
#define _TOOLS_H_

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <fftw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>

#include "Types.hpp"

using namespace std;
using namespace Eigen;

//! Collection of useful functions.
namespace Tools{
  template <class T> void setConstant(T* array, size_t N, T val);
  template <class T> void setZero(T* array, size_t N);
  template <class T> void copyarray(const T *A, T *B, size_t N);
  template <class T> void truncarray(T *X, size_t N1, size_t N2);
  void setZeroComplex(fftw_complex * array, size_t N);

  double *float2double(const float *, size_t);
  void float2double(const float *, double *, size_t);
  void double2float(const double *, float *, size_t);
  
  ArrayXi subseq(int L0, int N, bool endpoint=false);
  ArrayXd sign(const ArrayXd &);
  // ArrayXXd colwise_prod(const ArrayXXd &A, const ArrayXd &v);
  // ArrayXXd rowwise_prod(const ArrayXXd &A, const ArrayXd &v);
  ArrayXd cumsum(const ArrayXd &A);
  // void mat_diag_inplace(ArrayXXd &A, const ArrayXd &D);
  // void diag_mat_inplace(const ArrayXd &D, ArrayXXd &A);
  // ArrayXd rowwise_normalize_inplace(ArrayXXd &A);

  size_t find_idx_eq(const size_t *A, size_t M, size_t x, size_t *Idx);
  size_t find_idx_eq(const ArrayXu &A, size_t x, ArrayXu &Idx);
  size_t find_idx_leq(const ArrayXu& A, size_t x, ArrayXu &Idx);
  // bool dequal(double a, double b);
  double relative_error(double a, double b);

  Array2d nodeIdx2pos(int nodeIdx, const Array2i &vshape, const Array2d &theta, double splStep);
  Array2d node2pos(int row, int col, const Array2d &theta, double splStep);
  Array2d node2pos(int row, int col, const Array2i &vshape, const Array2d &theta, double splStep);
  Array2d pix2pos(int row, int col, const Array2i &vshape, double pixSize);
  Array2d pix2pos(int row, int col, size_t rows, size_t cols, double pixSize);
  Array2i pos2pix(const Array2d &pos, double spObj, size_t rows, size_t cols);
  Array2i pos2pix(const Array2d &pos, double spObj, const Array2d &sizeObj);

  double dist_gaussian(double a, double b, double sigma);
  double modulo(double a, double b);

  // Image quality assessment
  template<class T>
  double SNR(const T &X, const T &Y);
  template<class T>
  double Corr(const T &X, const T &Y);
  template<class T>
  double UQI(const T &X, const T &Y);
  double StreakIndex(const ArrayXd &X, const ArrayXd &im, const Array2i &dim);
  double StreakIndex(const ImageXXd &X, const ImageXXd &im);

  ArrayXu Photon(const ArrayXd &, size_t);
  ArrayXd LogPhoton(const ArrayXu &B, size_t phs);
  ArrayXd poisson_data(const ArrayXd &Y, size_t phs, Array3d &info);
  ArrayXd gaussian_data(const ArrayXd &Y, double snr, Array3d &info);
  double sg_noiselevel(const ArrayXd Y, int nbProj, int pixDet);

  // ArrayXd fftshift_2d(const ArrayXd &, size_t, size_t);
  // void fftshift_2d(const fftw_complex *X, size_t row, size_t col, fftw_complex *Y);
  // ArrayXd zeropadding(const ArrayXd &, const Array2i &, const Array2i &);
  // double abscomplex(fftw_complex);
  // double square_abscomplex(fftw_complex);

  void zoomin_fft(const double *X0, const Array2i &vshape, double *Y0, size_t upSpl, int method, double *W);
  void zoomin_interpl(const double *X0, const Array2i &vshape, double *Y0, size_t upSpl, int method);

  void cudaSafe(cudaError_t error, const string &message);
  void test_device(int);
  void setActiveGPU(int=-1);

  string itoa_fixlen3(int n);
  double log_factorial(size_t N);

  template<class T>
  T multi_imsum(const vector <T> &X);

  template<class T>
  vector<T> multi_imcumsum(const vector <T> &X);

  ImageXXd resampling_piximg(const double *X, int rowX, int colX, int rowY, int colY, double pixSizeY, bool oserr);
  ImageXXd resampling_piximg(const double *X, const Array2i &dimObjX, const Array2i &dimObjY, double pixSizeY, bool oserr);
  ImageXXd resampling_piximg(const double *X, const Array2i &dimObjX, int dim, bool oserr);
  ImageXXd resampling_piximg(const double *X, int rowX, int colX, int nrow, int ncol, bool oserr);
  //  ImageXXd resampling_evenprop(const double *X, const Array2i &dimObjX, int dim);

  ArrayXi random_permutation(size_t N, size_t M);
  ArrayXd joint(const vector<ArrayXd> &Y);
  vector<ArrayXd> separate(const ArrayXd &X, vector<ArrayXd> &Y) ;

  template <class T>
  T Nth_maxCoeff(const T *X, size_t M, size_t N) ;

  int l0norm(const ArrayXd &X);
}

#endif
