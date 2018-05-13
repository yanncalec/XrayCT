#ifndef _BLOBINTERPL_H_
#define _BLOBINTERPL_H_

#include <string>
#include <iostream>
#include <Eigen/Core>

//#include "Tools.hpp"
//#include "Blob.hpp"
//#include "Grid.hpp"
#include "Types.hpp"
#include "LinOp.hpp"
#include "BlobImage.hpp"
#include "GPUKernel.hpp"

using namespace Eigen;
using namespace std;


//! Class for interpolation operators (e.g., screen display, gradient, etc.) of a blob image. 
class BlobInterpl : public BlobImage, public LinOp {
protected :
  vector<double * > d_X;	//!< Device memory for blob coefficients
  vector<double * > d_Y;	//!< Device memory for interpolation grid coefficients
  ArrayXu memsize_X;		//!< Memory size for each d_X
  size_t memsize_Y;		//!< Memory size for d_Y
  double *sY; //!< temporary memory for the summation of individual interpolation

  LinOpType linoptypeT; //!< Adjoint interpolation operator's type.
  void BlobInterpl_init();

public :
  const Grid *igrid;		//!< interpolation grid is the same for all scales.

  BlobInterpl(const BlobImage *BI, //!< BlobImage object
	      const Grid *igrid,   //!< Interpolation grid
	      LinOpType linoptype)   //!< Interpolation operator type 
    : BlobImage(*BI), LinOp(0, this->nbNode, linoptype), igrid(igrid)
  { this->BlobInterpl_init(); }

  ~BlobInterpl();

  void _forward(const double *X, double *Y) ;
  void _backward(const double *X, double *Y) ;

  friend ostream& operator<<(ostream&, const BlobInterpl&);  
};


#endif
