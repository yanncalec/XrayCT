#ifndef _SPLINOP_H_
#define _SPLINOP_H_

#include <string>
#include <iostream>
#include <Eigen/Core>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_wavelet.h>
#include <gsl/gsl_wavelet2d.h>
#include <gsl/gsl_errno.h>

#include "LinOp.hpp"

using namespace Eigen;
using namespace std;

//! Gradient operator on pixel image
class PixGrad : public LinOp {
protected :
  Array2i dimObj;		//!< Dimension of pixel image. dimObj.x (or dimObj[0]) = cols, dimObj.y (or dimObj[1] = rows)
  //LinOpType linoptypeT; //!< Adjoint interpolation operator's type.
  int mode;			//!< Boundary mode : 1 for periodization, 0 for mirror symmetry
  ImageXXd Xh;
  ImageXXd Xv;

public :
  PixGrad(const Array2i &dimObj, int mode=0);

  ~PixGrad() {};

  void _forward(const double *X, double *Y) ;
  void _backward(const double *X, double *Y) ;

  vector<ImageXXd> imgrad(const ImageXXd &) ; //!< Calculate the gradient of an image
  ImageXXd imgradT(const vector<ImageXXd> &) ; //!< adjoint (like a laplacian)

  friend ostream& operator<<(ostream&, const PixGrad&);  
};


//! 2D Wavelet transform based on GSL implementation.
/*!  GSL takes only squared image of size \f$2^n\times 2^n\f$, and
  the transforms are in-place. Here we use embedding and restriction
  methods to work with non-squared image. This is a synthesis
  operator, which means that forward() takes wvl coeffs and returns a
  pixel image, while backward() takes a pixel image and returns the
  wvl coeffs. Consequently, dimX here is the dimension of wvl
  coefficient, and dimY is the pixel image dimension */

class GSL_Wavelet2D : public LinOp {
protected:
  gsl_wavelet *wvl;		 //!< GSL Wavelet object
  gsl_wavelet_workspace *wvl_ws; //!< GSL workspace pointer
  int nbScales;			 //!< Number of scales in wvl decomposition
  int gsl_dim;			//!< Squared image dimension for GSL (gsl_dim=N, NXN is the augmented squared image dimension)
  bool dimdiff;			//!< indicator for the dimension different to the GSL standard  \f$2^n\times 2^n\f$
  double *gsl_data;		//!< Temporary memory for GSL transform in case that LinOp dimension <> gsl squared dimension

  void transform(const double *X, double *Y, int dir);

public:
  GSL_Wavelet2D(const Array2i &dim, //!< pixel image dimension on which applies the wvl transform
		const string &wvlname, //!< name of GSL wavelet
		int order=0	       //!< order of GSL wavelet
		);

  ~GSL_Wavelet2D();

  int get_nbScale() {
    return this->nbScales;
  }

  void _forward(const double *X, double *Y) {
    this->transform(X, Y, 1);
  }

  void _backward(const double *X, double *Y) {
    this->transform(X, Y, -1);
  }

  string get_wvlname() {
    return string(gsl_wavelet_name(this->wvl));
  }

};


#endif
