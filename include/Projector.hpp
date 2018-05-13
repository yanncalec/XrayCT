#ifndef _PROJECTOR_H_
#define _PROJECTOR_H_

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

//! Abstract class of X-ray projector
/*!  No image model is included. It is possible to change the active
  index set of projections, which is useful in Ordered Subset
  algorithms (e.g., OSEM, OSART). Although GPU oriented, from this
  class we can derive the sparse matrix based projector class.
 */

class Projector : public AcqConfig {
protected :
  //! Initialize projectors, i.e., creat the sparse matrix or allocate GPU memory
  virtual void init_projector() = 0; //{ cout<<"Projector::init_projector not implemented!"<<endl; throw 0; } 
  //! Update the implicated parameters after change this->projset
  virtual void update_projector() = 0; //{ cout<<"Projector::update_projector not implemented!"<<endl; throw 0; }
  //! The mask for active detector bins. If maskDet[i]=true, then the i-th detector bin (wrt all projections) is active. Useful in case of dead detector pixels or the random mask in Compressed Sensing.
  ArrayXb maskDet;
  
public :
  //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int nbProj;		    //!< number of active projections
  ArrayXi projset;		//!< Subset of active of projections. Source of this set is active. Useful for OS(Ordered subset) algorithm.

  Projector() : AcqConfig(), nbProj(0), projset() { maskDet.setConstant(nbProj*pixDet, true); };
  Projector(const AcqConfig &);

  int get_nbProj() { return this->nbProj; } //!< return the number of active projections
  void set_nbProj(int = 0);	//!< set the number of active projections by choosing in a equally distributed manner the source position. 

  void set_projset(const ArrayXi &); //!< set manually the active projection index set.
  void reset_projset();		     //!< reset the active projection set (to all projections).
  void set_maskDet(const ArrayXb &); //!< set manually the mask for active detector bins.
  void reset_maskDet();		     //!< reset the active detector bins (to all bins).

  friend ostream& operator<<(ostream& out, const Projector& P);
};


//! GPU Projector class for blob image, with acquisition related informations preloaded on GPU.
/*!  The design principle here is to separate the acquisition
  information and the blob integration table (rarely used in
  practice), in order to share memory. All the member variables with
  the name starting by "d_" is the GPU device memory of the CPU
  counterpart.
*/

class GPUProjector : public Projector {

protected :
  int *d_projset;		//!< active projection set, device memory 
  double *d_rSrc;
  double *d_pSrc;
  double *d_rDet;
  //  double *d_rtDet;
  bool *d_maskDet;

  void init_projector(); //!< upload the parameters to GPU and retrieve the pointers *d_XX
  void update_projector();	//!< update the GPU memory after modifying the active projection set.

public :
  GPUProjector(const AcqConfig &conf);
  ~GPUProjector();

  friend ostream& operator<<(ostream&, const GPUProjector&);
  friend class BlobProjector;
};


//! Pixel-driven projector
class PixDrvProjector : public GPUProjector, public LinOp {
private :
  double *d_X;			//!< Device memory for X, input vector
  double *d_Y;			//!< Device memory for Y , output vector
  size_t memsize_X;		//!< size of device memory for X
  size_t memsize_Y;		//!< size of device memory for Y

  void PixDrvProjector_init();	//!< Memory allocation and upload etc.

public :
  const Array2i dimObj;		//!< Pixel image dimension. dimObj.x : number of columns
  const double spObj;		//!< side size of square pixel
  const Array2d sizeObj;	//!< Physical size of rectangular object. sizeObj.x : width, sizeObj.y : height

public :
  PixDrvProjector(const AcqConfig &conf, const Array2i &dimObj, double spObj) 
    : GPUProjector(conf), LinOp(conf.pixDet * conf.nbProj_total, dimObj, _Proj_), 
      dimObj(dimObj), spObj(spObj), sizeObj(dimObj.x() * spObj, dimObj.y() * spObj)
  { PixDrvProjector_init(); }

  ~PixDrvProjector();

  void _forward(const double *X, double *Y) ;
  void _backward(const double *X, double *Y) ;

  friend ostream& operator<<(ostream&, const PixDrvProjector&);  
};

#endif
