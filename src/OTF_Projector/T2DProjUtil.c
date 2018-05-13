#include "T2DProjector.h"
/* Useful functions for 2D projector */

T2DVector intersection(T2DVector S, T2DVector ds, T2DVector T, T2DVector dt)
/* Calculate the intersection point of segment starting from S with direction ds
   and the segment starting from T with direction dt
 */
{
  T2DVector ST = minus(T, S);
  double l = cross(ST, dt) / cross(ds, dt);
  return apcb(S, l, ds);
}

T2DVector pix2pos(int row, int col, double spObj, double sizeObjx, double sizeObjy)
/* Conversion from pixel to cartesian position, image is centered, with left-up corner the (0,0) pixel */
{
  T2DVector P;		/* Pixel position */
  P.x=(col+0.5)*spObj - sizeObjx/2;
  P.y=sizeObjy/2 - (row+0.5)*spObj;
  //printf("P : (%d,%d), (%f,%f), pixObj=%d, dh=%f\n",row,col,P.x,P.y,pixObj,dh);
  return P;
}

T2DPixel pos2pix(double x, double y, double spObj, double sizeObjx, double sizeObjy)
/* Conversion from cartesian position to pixel, image is centered, with left-up corner the (0,0) pixel */
{
  int dimObjr, dimObjc;		/* Object's resolution in pixel row and col */
  dimObjr = (int)floor(sizeObjy / spObj); /* sizeObjy is the height(y-axis) */
  dimObjc = (int)floor(sizeObjx / spObj); /* sizeObjx is the width(x-axis) */
  T2DPixel pp;  
  int cc = (int)trunc((x+sizeObjx/2)/spObj);
  int rr = -(int)trunc((y-sizeObjy/2)/spObj);

  rr=(rr<0)?0:rr;
  rr=(rr>dimObjr-1)?(dimObjr-1):rr;
  cc=(cc<0)?0:cc;
  cc=(cc>dimObjc-1)?(dimObjc-1):cc;
  pp.col = cc;
  pp.row = rr;
  return pp;
}

double modulo(double a, double b)
{
  double q=trunc(a/b);
  return a-q*b;
}

double dist_gaussian(double a, double b, double sigma) //P(X \in (a,b)) sigma>0
{ 
  return fabs(erf(b/sqrt(2)/sigma)-erf(a/sqrt(2)/sigma))/2;
}

double norm(T2DVector A)
{
  return sqrt(A.x*A.x+A.y*A.y);
}

T2DVector normalize(T2DVector A)
{
  T2DVector B;
  double nA = norm(A);
  B.x = A.x/nA;
  B.y = A.y/nA;
  return B;
}

T2DVector rotation(T2DVector A, double theta)
/* Anti-clockwise rotation of A by theta */
{
  T2DVector B;
  B.x = cos(theta)*A.x - sin(theta)*A.y; 
  B.y = sin(theta)*A.x + cos(theta)*A.y; 
  return B;
}

T2DVector apcb(T2DVector A, double c, T2DVector B)
/* return A+c*B */
{
  T2DVector S;
  S.x = A.x+c*B.x;
  S.y = A.y+c*B.y;
  return S;
}

T2DVector sum(T2DVector A, T2DVector B)
{
  T2DVector S;
  S.x = A.x+B.x;
  S.y = A.y+B.y;
  return S;
}

T2DVector minus(T2DVector A, T2DVector B)
/* return the vector from B to A : BA */
{
  T2DVector S;
  S.x = A.x-B.x;
  S.y = A.y-B.y;
  return S;
}

double cross(T2DVector A, T2DVector B)
/* return the determinant formed by |A B| */
{
  return A.x*B.y - A.y*B.x;
}

double dot(T2DVector A, T2DVector B)
{
  return A.x*B.x+A.y*B.y;
}

int qsort_comp_func_ascent(const void *a,const void *b)
{
  double *arg1 = (double *) a;
  double *arg2 = (double *) b;
  if( *arg1 < *arg2 ) return -1;
  else if( *arg1 == *arg2 ) return 0;
  else return 1;
}

int qsort_comp_func_descent(const void *a,const void *b)
{
  double *arg1 = (double *) a;
  double *arg2 = (double *) b;
  if( *arg1 > *arg2 ) return -1;
  else if( *arg1 == *arg2 ) return 0;
  else return 1;
}

int dequal(double a, double b)
{
  if (islessequal(a,b) && islessequal(b,a))
    return 1;
  else
    return 0;
/*   if (fabs((a-b)/b) < Epsilon) */
/*     return 1; */
/*   else */
/*     return 0; */
}

double relative_error(double a, double b){
  return fabs(a-b)/b;
}

void set_constant(double *X, int N, double v) {
  for (int n=0; n<N; n++)
    X[n] = v;
}

double lookuptable(double *T, int N, double lambda, double v)
{
  /* double *T : pre-integrated footprint table of standard blob of
     radius 1., the values in table T correspond the \int_{-\infty}^t
     f(x)dx */  
  /* int N : resolution of footprint, i.e. the length of footprint table */
  /* double lambda : dilation factor phi_lambda(x) = phi(lambda.x) */
  /* double v : the value to be looked up in the table */
  /* Important relation : Let phi_lambda(x) = phi(lambda.x), then
     1. Abel phi_lambda(t) = lambda^-1 Abel phi(\lambda t)
     2. Strip integral : \int_{-\infty}^B Abel phi_lambda(t) dt = lambda^-2 \int_{-\infty}^{lambda B} Abel phi(t) dt
   */

  double val;
  //double dt=2/N; /* the spacing size of T */

  v *= lambda;
  int idx=(int)floorf((v + 1) * N / 2);	/* v is the signed distance to projection profile center*/
  if (idx<0)
    val = 0;
  else if (idx >= N-1)
    val = T[N-1];
  else{
    double s = fabsf((v + 1) * N/ 2 - idx); /* residual to idx */
    val = (1-s)*T[idx] + s*T[idx+1]; /* Linear interpolation */
  }
  return val / (lambda*lambda);
}

