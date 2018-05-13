#ifndef _LINSOLVER_H_
#define _LINSOLVER_H_

#include "LinOp.hpp"

using namespace Eigen;
using namespace std;

namespace LinSolver{
  //! Solver for the normal equation \f$ A^\top A x = y \f$, using Conjugated Gradient method
  ConvMsg Solver_CG(LinOp &AtA,	//!< Symmetric system operator (e.g. a AtAOp object)
		    const ArrayXd &Y, //!< data vector Y
		    ArrayXd &X,	      //!< initialization and output
		    int maxIter=1000, //!< maximum number of CG iteration
		    double tol=1e-4,	 //!< stopping error tolerance
		    bool verbose = false
		    );

  //! CG solver for AX=Y, obselete version
  ConvMsg Solver_normal_CG(LinOp &A, //!< (non) symmetric system operator (not a AtAOp object)
			   const ArrayXd &Y, //!< data vector Y
			   ArrayXd &X,	      //!< initialization and output
			   int maxIter=1000, //!< maximum number of CG iteration
			   double tol=1e-4,	 //!< stopping error tolerance
			   const ArrayXi *Mask0=NULL, //!< Mask for X
			   bool verbose = false
			   );

  ConvMsg Projector_Debiasing(LinOp &A, const ArrayXd &Y, ArrayXd &X, 
			      int maxIter=100, double tol=1e-4, bool verbose = false);

}

#endif 
