#ifndef _LINOP_H_
#define _LINOP_H_

#include <string>
#include <iostream>
#include <Eigen/Core>

#include "Tools.hpp"
//#include "Blob.hpp"
//#include "Grid.hpp"
#include "Types.hpp"

using namespace Eigen;
using namespace std;


//! Linear Operator class
class LinOp {
protected :
  int dimY;	       //!< Dimension of image vector
  int dimX;	       //!< Dimension of domain vector
  
  Array2i shapeY;	//!< Shape of image, col(x) x row(y)
  Array2i shapeX;	//!< Shape of domain (length of 1D signal, col(x) x row(y) of 2D image)

  string linopname;		//!< Name of linear operator
  LinOpType linoptype;		//!< Known type of linear operator

public :
  //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  LinOp() : dimY(0), dimX(0), shapeY(), shapeX(), linopname(), linoptype(_UnknownOp_) { }  

  //! Constructor given the dimension of Y and X 
  LinOp(int dimY, int dimX, LinOpType linoptype=_UnknownOp_) 
    : dimY(dimY), dimX(dimX), shapeY(1,dimY), shapeX(1,dimX), linoptype(linoptype) {}

  //! Constructor given the shape of Y and X 
  LinOp(const Array2i &shapeY, const Array2i &shapeX, LinOpType linoptype=_UnknownOp_)
    : dimY(shapeY.prod()), dimX(shapeX.prod()), shapeY(shapeY), shapeX(shapeX), linoptype(linoptype) {}
  // {
  //   assert(this->shapeX.x() >= 0 && this->shapeX.y() >= 0); 
  //   assert(this->shapeY.x() >= 0 && this->shapeY.y() >= 0); 
  // }

  //! Constructor given the shape of Y and dimension of X 
  LinOp(const Array2i &shapeY, int dimX, LinOpType linoptype=_UnknownOp_)
    : dimY(shapeY.prod()), dimX(dimX), shapeY(shapeY), shapeX(1,dimX), linoptype(linoptype) {}
  // {
  //   assert(this->shapeY.x() >= 0 && this->shapeY.y() >= 0); 
  //   assert(this->dimX >= 0);
  // }
      
  //! Constructor given the dimension of Y and shape of X 
  LinOp(int dimY, const Array2i &shapeX, LinOpType linoptype = _UnknownOp_)
    : dimY(dimY), dimX(shapeX.prod()), shapeY(1,dimY), shapeX(shapeX), linoptype(linoptype) {}

  //! Copy constructor
  LinOp(const LinOp &A) : 
    dimY(A.dimY), dimX(A.dimX), shapeY(A.shapeY), shapeX(A.shapeX), linopname(A.linopname), linoptype(A.linoptype) {};

  //! Estimate the two-norm of current linear operator
  virtual double estimate_opNorm();

  // //! Set the name of current linear operator
  // void set_linopname(const string &name) { this->linopname = name; }

  //! Set the type of current linear operator
  void set_linoptype(LinOpType linoptype);

  //! Set the shape of domain vector
  void set_shapeX(int, int=1);
  //! Set the shape of image vector
  void set_shapeY(int, int=1);

  int get_dimX() const { return this->dimX; }
  int get_dimY() const { return this->dimY; }
  // int get_rankX() const { return this->shapeX.rank(); }
  // int get_rankY() const { return this->shapeY.rank(); }
  Array2i get_shapeX() const { return this->shapeX; }
  Array2i get_shapeY() const { return this->shapeY; }
  string get_linopname() const { return this->linopname; }
  LinOpType get_linoptype() const { return this->linoptype; }

  //! Forward operator, applying on double array. User must take care to initialize (to zero) the output vector.
  virtual void _forward(const double *, double *) = 0;
  //! Backward operator, applying on double array.  User must take care to initialize (to zero) the output vector.
  virtual void _backward(const double *, double *) = 0;

  //! Wrapper for forward operator, inplace. Output array is NOT initialized.
  void forward(const ArrayXd &, ArrayXd &) ;
  //! Wrapper for backward operator, inplace. Output array is NOT initialized.
  void backward(const ArrayXd &, ArrayXd &) ;

  //! Wrapper for forward operator. Output array is NOT initialized.
  ArrayXd forward(const ArrayXd &) ;
  //! Wrapper for backward operator. Output array is NOT initialized.
  ArrayXd backward(const ArrayXd &) ;
  //! Forward followed by the backward operator, i.e. \f$ A^\top A \f$
  //ArrayXd backforward(const ArrayXd &) ; 

  friend ostream& operator<<(ostream& out, const LinOp& B);
};

//! Symmetric operator: \f$ A^\top = A \f$
class SymmOp : public LinOp {
public:
  SymmOp(int dimX) : LinOp(dimX, dimX) {}

  void _backward(const double *X, double *Y) {
    this->_forward(X, Y);
  }
};


//! Identity operator
class IdentOp : public SymmOp {
protected:
public:
  IdentOp(int dimX) : SymmOp(dimX) {}

  void _forward(const double *X, double *Y);
};


// class MultiplyOp : public SymmOp {
//   // The operator A multiplied by a constant a
// protected:
//   double a;

// public:
//   MultiplyOp(int dimX, double a) : SymmOp(dimX), a(a) {}

//   void set_a(double a) {
//     this->a = a;
//   }

//   void _forward(const double *X, double *Y);
// };

//! Diagonal operator : multiply a vector elementwisely by a diagonal matrix
class DiagOp : public SymmOp {
protected:
  ArrayXd a;

public:
  DiagOp(const ArrayXd &a) : SymmOp(a.size()), a(a) {}

  DiagOp(int dimX, const ArrayXd &a) : SymmOp(dimX), a(a) { assert(a.size()==dimX); }

  DiagOp(int dimX, double cst) : SymmOp(dimX) { 
    this->a = ArrayXd::Constant(dimX, cst); 
  }

  void set_a(const ArrayXd a);

  void _forward(const double *X, double *Y);
};


//! The composition linear operator: \f$ A^\top A \f$, with \f$ A \f$ arbitrary linear operator here.
class AtAOp : public SymmOp {
protected:
  LinOp *A;
  double *toto;

public:
  AtAOp(LinOp *A) : SymmOp(A->get_dimX()), A(A)
  {
    toto = new double[A->get_dimY()];
  }

  ~AtAOp() {
    delete [] toto;
  }

  void _forward(const double *X, double *Y) ;
};

//! The composition linear operator: \f$A^\top D(a) A \f$with arbitrary linear operator \f$A\f$, \f$D(a)\f$ a diagonal matrix made from vector \f$a\f$
class AtDAOp : public SymmOp {
protected:
  LinOp *A;
  ArrayXd a;
  double *toto;

public:
  AtDAOp(LinOp *A, const ArrayXd &a) : SymmOp(A->get_dimX()), A(A), a(a)
  {
    assert(a.size() == A->get_dimY());
    toto = new double[A->get_dimY()];
  }

  void set_a(const ArrayXd &);

  ~AtDAOp() {
    delete [] toto;
  }

  void _forward(const double *X, double *Y) ;
};

//! Composition of two linear operators \f$A\circ B\f$: AB.forward(x) = A(B(x))
class CompOp : public LinOp {
protected:
  LinOp *A;
  LinOp *B;

  double *toto;

public:
  CompOp(LinOp *A, LinOp *B);

  ~CompOp() { 
    delete [] toto; 
  }

  void _forward(const double *X, double *Y) ;

  void _backward(const double *X, double *Y);
};

//! Composition of two linear operators \f$aA+bB\f$ with A, B arbitrary linear operators, and a,b two constants
class PlusOp : public LinOp {
protected:
  LinOp *A;
  LinOp *B;

  double a;
  double b;
  double *totoA;
  double *totoB;
  double *totoAt;
  double *totoBt;

public:
  PlusOp(LinOp *A, LinOp *B, double a, double b);

  ~PlusOp();

  void set_a(double a) { this->a = a; }
  void set_b(double b) { this->b = b; }

  void _forward(const double *X, double *Y) ;

  void _backward(const double *X, double *Y);
};


// class LinOpComp_SymPlus : public LinOp {
//   // The composition linear operator: A^T D(a) A + B^T D(b) B
//   // with A, B arbitrary linear operators,
//   // D(a), D(b) two diagonal matrices made from vectors a and b

// private:
//   LinOp *A;
//   LinOp *B;
//   ArrayXd a;
//   ArrayXd b;

//   ArrayXd totoA;
//   ArrayXd totoB;
//   double *Z;

// public:
//   LinOpComp_SymPlus(LinOp *A, LinOp *B, const ArrayXd * =NULL, const ArrayXd * =NULL);
//   ~LinOpComp_SymPlus() { delete [] Z; }

//   void set_a(const ArrayXd * =NULL);
//   void set_b(const ArrayXd * =NULL);

//   void _forward(const double *X, double *Y) ;
//   void _backward(const double *X, double *Y) ;
// };


// class LinOp_GMRF : public LinOp {
// private:
//   LinOp *A;
//   LinOp *B;
//   double reg_Id;
//   double reg_Lap;

//   ArrayXd totoA;
//   ArrayXd totoB;
//   double *Z;

// public:
//   LinOp_GMRF(LinOp *A, LinOp *B, double, double);
//   ~LinOp_GMRF() { delete [] Z; }

//   void set_Id(double a) { reg_Id=a; }
//   void set_Lap(double a) { reg_Lap=a; }

//   void _forward(const double *X, double *Y) ;
//   void _backward(const double *X, double *Y) ;
// };

#endif
