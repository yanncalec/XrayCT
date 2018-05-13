#include "LinOp.hpp"

// Member functions of class LinOp
void LinOp::forward(const ArrayXd &X, ArrayXd &Y) 
{ 
#ifdef _DEBUG_
  assert(X.size() == this->dimX && Y.size() == this->dimY);
#endif
  
  this->_forward(X.data(), Y.data());

  // if (precond == 1)
  //   Y /= this->opNorm;
  // if (this->usemask)
  //   Y *= this->mask;
}

void LinOp::backward(const ArrayXd &Y, ArrayXd &X) 
{     
#ifdef _DEBUG_
  assert(X.size() == this->dimX && Y.size() == this->dimY);
#endif

  this->_backward(Y.data(), X.data());
  // if (precond == 1)
  //   X /= this->opNorm;
  //   this->_backward(Y.data(), X.data());
  // else if (precond == 1) {
  //   this->_backward((this->RowPrcd * Y).data(), X.data());
  // }
}

ArrayXd LinOp::forward(const ArrayXd &X) 
{ 
#ifdef _DEBUG_
  assert(X.size() == this->dimX);
#endif
  ArrayXd Y(this->dimY);
  this->_forward(X.data(), Y.data());
  // if (precond == 1)
  //   Y /= this->opNorm;
  return Y;

  // double *Y = new double[this->dimY];
  // this->_forward(X.data(), Y);
  // return Map<ArrayXd> (Y, this->dimY);
}

ArrayXd LinOp::backward(const ArrayXd &Y) 
{
#ifdef _DEBUG_
  assert(Y.size() == this->dimY);
#endif  
  ArrayXd X(this->dimX);  
  this->_backward(Y.data(), X.data());
  // if (precond == 1)
  //   X /= this->opNorm;
  return X;

  // double *X = new double[this->dimX];
  // this->_backward(Y.data(), X);
  // return Map<ArrayXd> (X, this->dimX);
}

// ArrayXd LinOp::backforward(const ArrayXd &X) 
// {     
//   ArrayXd X0(this->dimX);
//   ArrayXd Y(this->dimY);
//   this->forward(X, Y);
//   this->backward(Y, X0);
//   return X0;
//   // double *X0 = new double[this->dimX];
//   // double *Y = new double[this->dimY];
//   // this->_forward(X.data(), Y);  
//   // this->_backward(Y, X0);

//   // delete [] Y;
//   // return Map<ArrayXd> (X0, this->dimX);
// }

double LinOp::estimate_opNorm()
{
  // Estimation of spectral radius
  // |A|_2 <= sqrt(|A|_1 * |A|_inf), and |A|_1, |A|_inf norm can be evaluated easily

  int nbtest = 10;
  ArrayXd X(this->dimX); X.setZero();
  ArrayXd Y(this->dimY); Y.setZero();
  // int pd = this->precond;	// save state
  // this->precond=0;
  double R = 0;

  //Estimate the infinity norm
  int Idx=0;
  int incr = max((int)floor(this->dimY*1./nbtest), 1);
  double norminf = 0;

  for (int n=0; n<nbtest; n++) {
    Y.setZero();
    //X.setZero();
    Y[Idx] = 1;
    this->backward(Y, X);	// At this point no precond is set
    norminf = fmax(X.abs().sum(), norminf);
    Idx += incr;
  }
  
  //Estimate the one norm
  Idx=0;
  incr = max((int)floor(this->dimX*1./nbtest), 1);
  double normone=0;

  //Idx = Tools::random_permutation(this->dimX, nbtest);
  for (int n=0; n<nbtest; n++) {
    //Y.setZero();
    X.setZero();
    X[Idx] = 1;
    this->forward(X, Y);
    normone = fmax(Y.abs().sum(), normone);
    Idx += incr;
  }
  R = sqrt(normone * norminf);
  //printf("Estimated one, inf, and two norm : %e, %e, %e\n", normone, norminf, R);

  //this->precond = pd;		// restore
  return R;
}

// void LinOp::set_precond(int pd, int nbTest) 
// {
//   assert(pd>=0);
//   this->precond = pd;
//   this->opNorm = (pd) ? this->estimate_opNorm(nbTest) : 0;    
// }

// void LinOp::set_opNorm(double v) 
// {
//   assert(v>0);
//   this->precond = 1;
//   this->opNorm = v;    
// }

// double LinOp::get_opNorm() 
// {
//   if (this->precond == 0)
//     return 0;
//   else
//     return this->opNorm;    
// }

void LinOp::set_shapeX(int row, int col) 
{
  this->shapeX = Array2i(col, row);
  this->dimX = col * row;
}

void LinOp::set_shapeY(int row, int col)
{
  this->shapeY = Array2i(col, row);
  this->dimY = col * row;
}

void LinOp::set_linoptype(LinOpType linoptype) { 
  this->linoptype = linoptype;
  switch(linoptype) {
  case _Proj_: this->linopname="GPU Blob Projector"; break;
  case _Img_: this->linopname="GPU Blob Image Interpolator"; break;
  case _Grad_: this->linopname="GPU Blob Gradient Operator"; break;
  case _Derv2_: this->linopname="GPU Blob 2nd Derivative Operator"; break;
  case _UnknownOp_: this->linopname="Unknown Operator"; break;
  default : break;
  }
}

ostream& operator<<(ostream& out, const LinOp& B)
{
  out<<"----Linear Operator----"<<endl;
  out<<"Name : "<<B.linopname<<endl;
  out<<"Dimension of domain : "<<B.dimX<<endl;
  out<<"Shape of domain (R X C) : ("<<B.shapeX.y()<<", "<<B.shapeX.x()<<")"<<endl;
  out<<"Dimension of image : "<<B.dimY<<endl;
  out<<"Shape of image (R X C) : ("<<B.shapeY.y()<<", "<<B.shapeY.x()<<")"<<endl;
  return out;
}


// Member for class IdentOp
void IdentOp::_forward(const double *X, double *Y)
{
  for (int n=0; n<this->dimX; n++)
    Y[n] = X[n];
}

// // Member for class MultiplyOp
// void MultiplyOp::_forward(const double *X, double *Y)
// {
//   for (int n=0; n<this->dimX; n++)
//     Y[n] = this->a * X[n];
// }

// Member for class DiagOp
void DiagOp::set_a(const ArrayXd a) {
  assert(a.size() == this->dimX);
  this->a = a;
}

void DiagOp::_forward(const double *X, double *Y)
{
  for (int n=0; n<this->dimX; n++)
    Y[n] = this->a[n] * X[n];
}

// Member for class AtAOp
void AtAOp::_forward(const double *X, double *Y) 
{
  A->_forward(X, toto);
  A->_backward(toto, Y);
}

// Member for class AtDAOp
void AtDAOp::set_a(const ArrayXd &a)
{
  assert(a.size() == A->get_dimY());
  this->a = a;
}

void AtDAOp::_forward(const double *X, double *Y) 
{
  A->_forward(X, toto);
  for (int n=0; n<this->a.size(); n++) // Be careful, this->dimX is different of a.size()
    toto[n] *= this->a[n];

  A->_backward(toto, Y);
}

// Member for class CompOp
CompOp::CompOp(LinOp *A, LinOp *B) 
  : LinOp(A->get_dimY(), B->get_dimX()), A(A), B(B)
{
  assert(A->get_dimX() == B->get_dimY());
  toto = new double[A->get_dimX()];
}

void CompOp::_forward(const double *X, double *Y) 
{
  B->_forward(X, toto);
  A->_forward(toto, Y);
}

void CompOp::_backward(const double *X, double *Y) 
{
  A->_backward(X, toto);
  B->_backward(toto, Y);
}

// Member for class PlusOp
PlusOp::PlusOp(LinOp *A, LinOp *B, double a, double b) 
  : LinOp(A->get_dimY(), A->get_dimX()), A(A), B(B), a(a), b(b)
{
  assert(A->get_dimX() == B->get_dimX());
  assert(A->get_dimY() == B->get_dimY()); 

  totoA=new double[dimY];
  totoB=new double[dimY];
  totoAt=new double[dimX];
  totoBt=new double[dimX];
}

PlusOp::~PlusOp() { 
  delete [] totoA; 
  delete [] totoB; 
  delete [] totoAt; 
  delete [] totoBt; 
}

void PlusOp::_forward(const double *X, double *Y)
{
  if(abs(a)>0)
    A->_forward(X, totoA);
  if(abs(b)>0)
    B->_forward(X, totoB);

  for (int n=0; n<this->dimY; n++)
    Y[n] = a*totoA[n] + b*totoB[n];
}

void PlusOp::_backward(const double *X, double *Y)
{
  if(abs(a)>0)
    A->_backward(X, totoAt);
  if(abs(b)>0)
    B->_backward(X, totoBt);

  for (int n=0; n<this->dimX; n++)
    Y[n] = a*totoAt[n] + b*totoBt[n];
}
