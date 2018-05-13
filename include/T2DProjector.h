#ifdef _cplusplus
extern "C" {
#endif

#ifndef T2DPROJECTOR_H
#define T2DPROJECTOR_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#ifndef _EPSILON_
#define Epsilon pow(10, -8)
#endif

#ifndef _MAX_
#define max(a,b) ((a>b)?a:b)
#endif

#ifndef _MIN_
#define min(a,b) ((a<b)?a:b)
#endif

  //#define PRECISION 6 /* Precision trunk factor */
  //#define SIGMA 3 /* Oversampling factor */
  //#define KERNEL(r) (1.0-fabs((double)(r))/((double)R/2))

  typedef struct {
    double x;
    double y;
  } T2DVector;

  typedef struct {
    int row;
    int col;
  } T2DPixel;

  /* typedef struct { */
  /*   double rSrc; */
  /*   double rDet; */
  /*   double sizeObj; */
  /*   double sizeDet; */
  /*   int nbProj; */
  /*   int pixDet; */
  /*   int pixObj; */
  /*   double spos; */
  /*   int fanbeam; */
  /* } T2DAcqConfig; */

  /* T2DProjUtil.c */
  T2DVector intersection(T2DVector S, T2DVector ds, T2DVector T, T2DVector dt);
  T2DVector pix2pos(int row, int col, double spObj, double sizeObjx, double sizeObjy);
  T2DPixel pos2pix(double x, double y, double spObj, double sizeObjx, double sizeObjy);

  double modulo(double, double);
  double dist_gaussian(double a, double b, double sigma);

  double norm(T2DVector A);
  T2DVector normalize(T2DVector A);
  T2DVector rotation(T2DVector A, double theta);
  T2DVector apcb(T2DVector A, double c, T2DVector B);
  T2DVector sum(T2DVector A, T2DVector B);
  T2DVector minus(T2DVector A, T2DVector B);
  double cross(T2DVector A, T2DVector B);
  double dot(T2DVector, T2DVector );
  int qsort_comp_func_ascent(const void *,const void *);
  int qsort_comp_func_descent(const void *,const void *);
  int dequal(double a, double b);
  double relative_error(double, double);
  double lookuptable(double *T, int N, double lambda, double v);
  void set_constant(double *X, int N, double v);

  void BlobDriven_OTF(int fanbeam, 
		      double rSrc, 
		      double pSrc, 
		      double rDet, 
		      double rtDet,
		      double sizeDet, 
		      double spDet, 
		      int pixDet,
		      double radius,
		      double *T, int nT, 		       
		      double *Nodes_x, int nNodes_x, 
		      double *Nodes_y, int nNodes_y,
		      double *X, double *Y, int forward);

  void PixelDriven_OTF(int fanbeam, 
		       double rSrc, 
		       double pSrc, 
		       double rDet, 
		       double rtDet,
		       double spObj,
		       int pixObjr,
		       int pixObjc,
		       double spDet, 
		       int pixDet,
		       double *X, 
		       double *Y, 
		       int forward);

  void RegularSampling_OTF(int fanbeam, 
			   double rSrc, 
			   double pSrc, 
			   double rDet, 
			   double rtDet,
			   double spObj,
			   int pixObjr,
			   int pixObjc,
			   double spDet, 
			   int pixDet,
			   double *X, 
			   double *Y, 
			   int forward);

  void Siddon_OTF(int fanbeam, 
		  double rSrc, 
		  double pSrc, 
		  double rDet, 
		  double rtDet,
		  double spObj,
		  int pixObjr,
		  int pixObjc,
		  double spDet, 
		  int pixDet,
		  double *X, 
		  double *Y, 
		  int forward);

#endif

#ifdef _cplusplus
}
#endif
