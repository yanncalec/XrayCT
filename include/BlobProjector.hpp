#ifndef _BLOBPROJECTOR_H_
#define _BLOBPROJECTOR_H_

#include <string>
#include <iostream>
#include <Eigen/Core>

//#include "Tools.hpp"
//#include "Blob.hpp"
//#include "Grid.hpp"
#include "Projector.hpp"
#include "Types.hpp"
#include "LinOp.hpp"
#include "BlobImage.hpp"
#include "GPUKernel.hpp"

using namespace Eigen;
using namespace std;

//! Looking-up table for GPU projector. Useful only for strip-integral projector, or for the blob on which the on-the-fly evaluation is difficult (like Diff-Gaussian).
/*!  The table of blob can be either the abel transform or the
  integration of abel transform. Member variable stripint makes the
  distinction.
 */
class GPUProjTable {
private :
  ArrayXd T;			//!< Table, host memory
  double *d_T;			//!< device memory for T
  size_t dimT;		       //!< dimension of table, larger for a better precision. If blob radius is 0.5 and dimT=1000, then the precision of table is 1/1000.
  bool stripint; //!< Indicator of the table, true for strip integral table, false for direct abel table

public :
  GPUProjTable() : T(), d_T(0), dimT(0), stripint(false) {}
  GPUProjTable(const Blob &blob, bool stripint, size_t dimT);
  ~GPUProjTable();

  friend class GPUProjector;
  friend class BlobProjector;

  friend ostream& operator<<(ostream&, const GPUProjTable&);  
};

//! Blob image X-Ray projector.
class BlobProjector : public GPUProjector, public BlobImage, public LinOp {
private :
  vector<const GPUProjTable * > LT;	//!< Table for each scale of blob image

  vector<double * > d_X;			//!< Device memory for X, input vector
  vector<double * > d_Y;			//!< Device memory for Y , output vector

  double *sY;			//!< Host memory for temporary summation

  ArrayXu memsize_X;		//!< Memory size for X, scale by scale
  size_t memsize_Y;		

  vector<ArrayXb> maskBlob; //!< Mask for blob (by scale), if maskBlob[i] true, then the i-th blob is projectced/backprojected
  vector<bool * > d_maskBlob;	//!< Mask for blob, GPU memory

  void BlobProjector_init();

public :
  const bool tablemode;		//!< Indicator for looking-up table based projector. 0 for on-the-fly projection(no looking-up table, raytracing only), 1 : raytracing table(same as 0, but implemented through a table), 2 : strip integral table

  BlobProjector(const AcqConfig &conf, //!< Acquisition config
		const BlobImage *BI,   //!< BlobImage object
		int tablemode = 0,  //!< Looking-up table projector option
		size_t dimT = (size_t)pow(2, 14.) //!< size of table
		); 

  ~BlobProjector();

  void _forward(const double *X, double *Y) ;
  void _backward(const double *Y, double *X) ;

  void set_maskBlob(const vector<ArrayXb> &M);
  void reset_maskBlob();
  void update_projector();

  ArrayXd row_lpnorm(double lpnorm); //!< lp norm of each row (projector as a matrix), eg, p=0,1,2.. and p<0 for l_inf
  ArrayXd col_lpnorm(double lpnorm); //!< lp norm of each column (projector as a matrix)

  double estimate_opNorm();

  //void update_projector();	// modify the dimension of linop
  friend ostream& operator<<(ostream&, const BlobProjector&);  
};


#endif
