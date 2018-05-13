#ifndef _MARTOP_H_
#define _MARTOR_H_

#include "AcqConfig.hpp"
//#include "Blob.hpp"
#include "Tools.hpp"
#include "LinOp.hpp"
#include "GPUKernel.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <string>
#include <iostream>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

//! BackProjector for MART algorithm, obsolete
class MARTOp : public GPUProjector {
private :
  double *d_X;			//!< Device memory for X, pixel image
  double *d_Y;			//!< Device memory for Y , sinogram
  size_t memsize_X;		
  size_t memsize_Y;		
  void MARTOp_init();

public :
  const Array2d sizeObj;
  const Array2i dimObj;
  const double spObj;

public :
  MARTOp(const AcqConfig &conf, const Array2i &dimObj, double spObj) 
    : GPUProjector(conf), memsize_X(sizeof(double) * dimObj.prod()), 
      memsize_Y(sizeof(double)*conf.pixDet * conf.nbProj_total), 
      dimObj(dimObj), spObj(spObj), sizeObj(dimObj.x() * spObj, dimObj.y() * spObj)
  { MARTOp_init(); }

  ~MARTOp();

  void backward(const double *X, double *Y);
};

#endif
