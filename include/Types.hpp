#ifndef _TYPES_H_
#define _TYPES_H_

#include <Eigen/Core>
#include "GPUTypes.hpp"

using namespace Eigen;

#define BLOBPARMLEN 2		// Length of blob parameter set
#define STD_DIAMFOV 200		// Diameter of normalized FOV, the whole acquisition system is normalized to a standard size according to this value. 
// The reconstruction algorithms tunning parameter may depend on this value. Too small value cause numerical instability (think of exp(-alpha x*x), alpha depend on diamFOV).

typedef enum{_Cartesian_, _Hexagonal_, _Unknown_} GridType;

typedef Array<size_t, Dynamic, 1> ArrayXu;
typedef Array<size_t, 2, 1> Array2u;
typedef Array<bool, Dynamic, 1> ArrayXb;

typedef Array<double, Dynamic, Dynamic, ColMajor> ArrayXXd_ColMajor;
typedef Array<double, Dynamic, Dynamic, RowMajor> ImageXXd; // Row major for storing 2D image data

//! Class for convergence message of iterative algorithm. The senses of vobj, res and norm are context dependant.
struct ConvMsg {
  int niter;			//!< Number of iteration taken
  double vobj; 			//!< value of objective function
  double res;			//!< Residual error, refering normally to the L2 error \f$ \|Ax-b\| \f$. Remark that this value may be normalized in function of algorithm.
  double norm;			//!< Regularization norm value achieved by algorithm

  ConvMsg() : niter(0), vobj(0), res(0), norm(0) {}
  ConvMsg(int niter, double vobj, double res, double norm) : niter(niter), vobj(0), res(res), norm(norm) {}
  ConvMsg& operator =(const ConvMsg &msg) { 
    this->niter = msg.niter; 
    this->vobj = msg.vobj;
    this->res = msg.res;
    this->norm = msg.norm;
    return *this; 
  }
  //friend ofstream & operator <<(ofstream &out, const ConvMsg &msg);
};

#endif
