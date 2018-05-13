/*  -*- C -*-  */

/* Useful device functions for GPU calculation */

#ifndef _GPUDEVFUNC_H_
#define _GPUDEVFUNC_H_

/* __device__ inline double2 node2pos_centralized(int row, int col, int vrow, int vcol, double thetax, double thetay, double splStep) */
/* { */
/*   /\* Convertion from a 2d grid node index to its cordinate *\/ */
/*   /\* row, col : unsigned index (start from 0) of grid node *\/ */
/*   /\* thetax, thetay : second generating vector of grid, Important convention : theta is in IV octoplan *\/ */
/*   /\* splStep : sampling step of grid *\/ */
/*   /\* vrow, vcol : virtual dimension of grid in row and in column, must be even *\/ */
/*   /\* the first node (index 0) is the one at up-left corner of parallelogram formed by [[1,0]^t, theta] *\/ */

/*   return make_double2(splStep * ((col-vcol/2) + (row-vrow/2) * thetax), splStep * (row-vrow/2) * thetay); */
/* } */

__device__ inline double2 node2pos(int row, int col, int rows, int cols, double thetax, double thetay, double splStep)
{
  /* Convertion from a 2d grid node index to its cordinate */
  /* row, col : unsigned index (start from 0) of grid node */
  /* thetax, thetay : second generating vector of grid, Important convention : theta is in IV octoplan */
  /* splStep : sampling step of grid */
  /* rows, cols : virtual dimension of grid in row and in column, must be even */
  /* the first node (index 0) is the one at up-left corner of parallelogram formed by [[1,0]^t, theta] */

  return make_double2(splStep * ((col-cols/2) + (row-rows/2) * thetax), splStep * (row-rows/2) * thetay);
}

__device__ inline double2 node2pos(int row, int col, double thetax, double thetay, double splStep)
{
  // Calculate the position of a given node wrt the grid vector v1=[1,0]'
  // V = [v1, theta], sampling step
  return make_double2(splStep * (col + row * thetax), splStep * row * thetay);
}

__device__ inline double2 nodeIdx2pos(int nodeIdx, int rows, int cols, double thetax, double thetay, double splStep)
{
  /* Same as node2pos, but with the flatten node index */
  int rr = (int)floor(nodeIdx * 1. / cols);
  int cc = nodeIdx - rr * cols;

  return make_double2(splStep * (cc - cols/2 + (rr - rows/2) * thetax), splStep * (rr - rows/2) * thetay);
}


__device__ inline double2 pix2pos(int row, int col, int rows, int cols, double pixSize)
{
  /* Convert a pixel index to its coordinates */
  /* In a 2D plan, square and even dimension image is centered, the
     (0,0) pixel corresponds to the left-up corner */
  return make_double2((col + 0.5 - cols/2)* pixSize, (rows/2 - row - 0.5)* pixSize);
}

__device__ inline double2 pix2pos(int pixIdx, int rows, int cols, double pixSize)
{
  /* Same, with the flatten index */
  int rr = (int)floor(pixIdx * 1. / cols);
  int cc = pixIdx - rr * cols;

  return make_double2((cc + 0.5 - cols/2)* pixSize, (rows/2 - rr - 0.5)* pixSize);
}

/* Calculate the quotient and remainder of x, y */
__device__ inline double2 div(double x, double y)
{
  return make_double2(trunc(x / y), fmod(x,y));
}

__device__ inline double modulo(double a, double b)
{
  return a - trunc(a/b) * b;
}

/* Norm of a vector */
__device__ inline double norm(const double2 &A)
{
  return sqrt(A.x*A.x+A.y*A.y);
}

/* A plus B */
__device__ inline double2 operator +(const double2 &A, const double2 &B)
{
  return make_double2(A.x+B.x, A.y+B.y);
}

/* A minus B : vector from B to A */
__device__ inline double2 operator -(const double2 &A, const double2 &B)
{
  return make_double2(A.x-B.x, A.y-B.y);
}

/* c * A */
__device__ inline double2 operator *(double c, const double2 &A)
{
  return make_double2(c*A.x, c*A.y);
}

/* Dot multiplication */
__device__ inline double dot(const double2 &A, const double2 &B)
{
  return A.x*B.x+A.y*B.y;
}

/* Cross multiplication */
__device__ inline double cross(const double2 &A, const double2 &B)
/* return the determinant formed by |A B| */
{
  return A.x*B.y - A.y*B.x;
}

/* Rotation of a vector by theta */
__device__ inline double2 rotation(const double2 &A, double theta)
/* Anti-clockwise rotation of A by theta */
{
  return make_double2(cos(theta)*A.x - sin(theta)*A.y,
		     sin(theta)*A.x + cos(theta)*A.y);
}

__device__ inline double2 intersection(const double2 &S, const double2 &ds, const double2 &T, const double2 &dt)
/* Calculate the intersection point of segment starting from S with direction ds
   and the segment starting from T with direction dt. ds must be unitary vector.
 */
{
  double2 ST = T - S;
  double l = cross(ST, dt) / cross(ds, dt);
  return S + l * ds; //apcb(S, l, ds);
}

__device__ inline double2 cercle_projection(const double2 &S, const double2 &P, double radius, const double2 &L, const double2 &dU)
/* Calculate the projection of a cercle centered at P of radius,
   on the segment defined by L+ t*dU, by the ray starting from point S.
 */
{
  double2 SP = P-S;
  double2 SL = L-S;

  double sin_theta = radius / norm(SP); /* theta is the angle formed by the tangent vectors and SP, it's included in [0, pi/2] */
  double cos_theta = sqrt(1 - sin_theta*sin_theta);
	
  double2 SP1, SP2;	/* Tangent vectors with blob P */
  SP1.x = cos_theta * SP.x - sin_theta * SP.y; /* Rotation of SP by theta */
  SP1.y = sin_theta * SP.x + cos_theta * SP.y;
  SP2.x = cos_theta * SP.x + sin_theta * SP.y; /* Rotation of SP by -theta */
  SP2.y = -sin_theta * SP.x + cos_theta * SP.y;

  /* Arriving positions of SP1 SP2 on detector plan */
  double l1 = -cross(SP1, SL)/cross(SP1,dU); /* l1 is the distance on detector plan from L to the arriving point of SP1 */
  double l2 = -cross(SP2, SL)/cross(SP2,dU);
  return make_double2(fmin(l1,l2), fmax(l1,l2));
}

/* Atomic float add, slower than the official version atomicadd */
__device__ inline void atomicAddFloat(float* address, float value)
{
  float old = value;
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old)) != 0.0f) ;
  //return value;
}

/* Atomic double add, much slower than float version */
__device__ inline void atomicAddDouble(double *address, double value)  //See CUDA official forum
{
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
}

/* Atomic double max */
__device__ inline void atomicMaxDouble(double *address, double value)  //See CUDA official forum
{
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(fmax(__longlong_as_double(oldval), value));
  while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __double_as_longlong(fmax(__longlong_as_double(oldval),  value));
    }
}

/* Atomic double min */
__device__ inline void atomicMinDouble(double *address, double value)  //See CUDA official forum
{
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(fmin(__longlong_as_double(oldval), value));
  while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __double_as_longlong(fmin(__longlong_as_double(oldval),  value));
    }
}

/* Look-up table function */
__device__ inline double lookuptable(const double *T, int N, double v)
{
  /* double *T : strip-integral or direct footprint table of blob */  
  /* int N : resolution of footprint, i.e. the length of table */
  /* double v : the normalized value to be looked up in the table */

  double pos = (v + 1) / 2 * N;

  int idx=(int)floor(pos);	/* v is the signed distance to projection profile center*/

  if (idx<0)
    return 0;
  else if (idx >= N-1)
    return T[N-1];
  else{
    double s = fabs(pos - idx); /* residual to idx */
    return (1-s)*T[idx] + s*T[idx+1]; /* Linear interpolation */
  }
}

__device__ inline double log_factorial(size_t N)
{
    /* Approximation of log factorial by  */
    /* log n! ~ n(log n) - n + (log(n + 4 n^2 + 8 n^3))/6 + (log \pi)/2 */
    /* which is a better approximation (exp(log n!)) than Stirling formula. */
  double res;
  if (N>20) 
    res = N*log(N*1.) - N + log(1.*(N + 4*N*N + 8*N*N*N))/6 + log(M_PI)/2;
  else
    switch (N) {
      /* log(n!) for n=2..20 */
    case 0 : res = 0;
    case 1 : res = 0;
    case 2 : res = 0.69314718;
    case 3 : res = 1.79175947;
    case 4 : res = 3.17805383;
    case 5 : res = 4.78749174;
    case 6 : res = 6.57925121;   
    case 7 : res = 8.52516136;  
    case 8 : res = 10.6046029 ;  
    case 9 : res = 12.80182748;
    case 10 : res = 15.10441257;  
    case 11 : res = 17.50230785;  
    case 12 : res = 19.9872145 ;  
    case 13 : res = 22.55216385;
    case 14 : res = 25.19122118;  
    case 15 : res = 27.89927138;  
    case 16 : res = 30.67186011;  
    case 17 : res = 33.50507345;
    case 18 : res = 36.39544521;  
    case 19 : res = 39.33988419;  
    case 20 : res = 42.33561646;
    }

  /* double res = 0; */
  /* for (size_t n=1; n<N; n++) */
  /*   res += log(n); */
  return res;
}

#endif
