#ifndef _GRID_H_
#define _GRID_H_

#include "Types.hpp"
#include "Tools.hpp"

#include <fftw3.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

//! Class for 2D regular grid defined on a rectangular region.
/*!  
  \f$ g_1=[1, 0]^\top, g_2=[\cos\theta, \sin\theta]^\top \in
  S^1\f$, with \f$ -\pi/2 \leq \theta <0\f$ are two generating
  vectors. Given the sampling step $h$ : the grid nodes are \f$ h G k
  \f$, with \f$ k\in Z^2\f$, where the grid matrix \f$ G =[g_1, g_2]
  \f$.  
*/

class Grid {
private :
  int *d_mask;	      //!< Device memory for grid's active node mask
  int *d_Idx;      //!< Device memory for grid's active node index
  void init_grid(); //!< Initialization of grid, called by constructor  
  void upload_GPU();			//!< upload to device memory

protected:
  Array2i grid_vshape(const Array2d &, const Array2d &, double);
  void _embedding_complex_forward(const double *, fftw_complex *) const ;
  void _embedding_complex_backward(const fftw_complex *, double *) const ;
  void _embedding_forward(const double *, double *) const ; //<! embedding operator using indicator Idx and mask
  void _embedding_backward(const double *, double *) const ; //<! embedding operator using indicator Idx and mask

public :
  Array2d theta;		//!< unitary vector defining the grid
  Array2d sizeObj; //!< rectangular object object size : Width X Height (x-axis, y-axis)
  double diamROI;  //!< diameter of ROI
  double splStep;  //!< sampling step \f$ h \f$, this changes the grid nodes density.

  //! virtual shape of regular grid : row X column dimension.  
  //! The regular grid is linearly spanned by \f$ g_1 \f$ and \f$ g_2
  //! \f$ with integer coefficients.  vshape defines the ranges size
  //! of these integers.
  Array2i vshape; 
  
  //! index (C-order) in virtual 2d array (vshape) for
  //! active nodes, i.e. those bounded inside the object's
  //! support defined by sizeObj
  ArrayXi Idx;	

  ArrayXi mask;	//!< The mask for active nodes, mask[n]<0 means n-th node is inactive. We have mask[Idx[n]]==n
  int nbNode;		//!< Total number of active nodes
  const GridType gtype;		//!< Grid type 
  const string gridname;

public :
  //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Grid() : d_mask(0), theta(), sizeObj(), diamROI(0), splStep(0), vshape(), Idx(), mask(), nbNode(0), gtype(_Unknown_), gridname("") {}
  
  // copy constructor
  Grid(const Grid &grid) 
    : d_mask(0), theta(grid.theta), sizeObj(grid.sizeObj), diamROI(diamROI), splStep(grid.splStep), vshape(grid.vshape), 
    Idx(grid.Idx), mask(grid.mask), nbNode(grid.nbNode), gtype(grid.gtype), gridname(grid.gridname) 
  { this->upload_GPU(); }

  //! Constructors for hex or cartesian grid by their type
  //! Initialization from sizeObj, the virtual shape of grid (vshape) is automatically determined  
  Grid(const Array2d &sizeObj, double splStep, GridType gtype, double diamROI=0);

  //! Initialization from given virtual shape of grid, the object size (sizeObj) is automatically determined
  Grid(const Array2i *vshape, double splStep, GridType gtype, double diamROI=0);

  //! Initialization from given virtual shape of grid and the object size (sizeObj)
  //! this is useful in initializing the interpolation
  //! grid of a blobimage, which should be N times denser than blob grid.
  Grid(const Array2d &sizeObj, const Array2i &vshape, double splStep, GridType gtype, double diamROI=0);

  ~Grid();

  vector<ArrayXd> get_Nodes() const; //!< Generate the active nodes coordinates
  vector<ArrayXd> get_all_Nodes() const; //!< Generate all nodes coordinates

  ArrayXd embedding_forward(const ArrayXd &) const ;
  ArrayXd embedding_backward(const ArrayXd &) const ;

  double determinant() const; 	//!< Determinant of grid matrix \f$ \mbox{det} G \f$, with \f$ G= h [g_1, g_2] \f$
  void save(ofstream &) const; 	//!< Save a Blob object to output file stream
  static string type2str(GridType); //!< Convert a given grid type to a string 
  static Array2d type2theta(GridType gtype); //!< Return a generating vector from a given grid type 

  //template<class T> friend T & operator<<(T &out, const Grid &);
  friend ostream & operator<<(ostream &, const Grid &);
  friend ofstream & operator<<(ofstream &, const Grid &);

  friend class BlobImage;
  friend class MultiBlobImage;
  friend class BlobInterpl;
  friend class MultiBlobInterpl;
  friend class BlobProjector;
  friend class MultiBlobProjector;
};

#endif
