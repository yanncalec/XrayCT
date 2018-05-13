#include "T2DProjector.h"

/* 
Fan-Beam or Parallel-Beam Projection matrix by Regular Sampling Ray-Driven method 
*/

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
			 int forward)
{
  double sizeObjx = pixObjc * spObj;
  double sizeObjy = pixObjr * spObj;
  double sizeDet = pixDet * spDet; 

  T2DVector S; /* Source position */
  T2DVector D; /* Center of detector plan */
  D.x = rDet * cos(pSrc + M_PI); /* Center of detector plan is always at opposite side of source */
  D.y = rDet * sin(pSrc + M_PI);

  T2DVector dU;	 /* Unitary vector of the detector plan direction from
		   left border to right border, from source's viewpoint */
  dU.x = -sin(pSrc); dU.y = cos(pSrc); /* Without detector plan's
					  rotation, dU is the rotation
					  by pi/2 in anti-clockwise
					  sens of vector OS */
  dU = rotation(dU, rtDet);	/* dU of inclined detector plan */ 

  T2DVector L; /* Detector left-border position, from source's viewpoint */
  L = apcb(D, -sizeDet/2, dU); /* By definition, L = D - sizeDet/2 * dU, function apcb(A, c, B) gives A + c* B */

  T2DVector A;  /* In Point, near source */
  T2DVector B;  /* Out Point, near detector */

  double a1,a2,b1,b2;		/* Variables used in determination of entry and exit points */
  double amin, amax, bmin, bmax;
  double lA, lB;		/* A and B's parameter values */

  int SF = 2;		     /* Regular sampling factor */

  for (int dpix = 0; dpix < pixDet; dpix++){ /* dpix is the detector pixel index */
    /* Source position */
    if (fanbeam) { /* Fan-beam */
      S.x = cos(pSrc)*rSrc;
      S.y = sin(pSrc)*rSrc;
    }
    else { /* Parallel Beam */
      S.x = cos(pSrc)*rSrc-sin(pSrc)*(dpix+0.5-pixDet/2) * spDet;
      S.y = sin(pSrc)*rSrc+cos(pSrc)*(dpix+0.5-pixDet/2) * spDet;
    }
    
    T2DVector E = apcb(L, (dpix+0.5)*spDet, dU); /* Detector bin's position */

    int intersected=0;	      /* Bool variable of intersection flag */
    /* Calculate the intersection points with object borders : A and B */
    if (dequal(S.x, E.x)) { /* Source-Detector parallel to OY */
      A.x = (S.x + E.x) / 2;
      B.x = A.x;
      double cc = 2 * (double)(S.y > 0) - 1;
      A.y = cc * sizeObjy / 2;
      B.y = -cc * sizeObjy / 2;
      lA = (A.y - S.y) / (E.y - S.y);
      lB = (B.y - S.y) / (E.y - S.y);
      intersected = isless(fabs(A.x), sizeObjx/2); //(A.x > -sizeObjx/2) && (A.x < sizeObjx/2);
    }
    else if (dequal(S.y, E.y)){/* Source-Detector parallel to OX */
      //printf("(S.y, E.y) = (%2.10f, %2.10f)\n",S.y,E.y);
      A.y = (S.y+E.y)/2; 
      B.y = A.y;
      double cc = 2 * (double)(S.x > 0) - 1;
      A.x = cc * sizeObjx / 2;
      B.x = -cc * sizeObjx / 2;
      lA = (A.x - S.x) / (E.x - S.x);
      lB = (B.x - S.x) / (E.x - S.x);
      intersected = isless(fabs(A.y), sizeObjy/2); //      intersected = (A.y > -sizeObjy/2) && (A.y < sizeObjy/2);
    }
    else{
      a1=(sizeObjx/2 - S.x) / (E.x-S.x);
      a2=(-sizeObjx/2 - S.x) / (E.x-S.x);
      b1=(sizeObjy/2 - S.y) / (E.y-S.y);
      b2=(-sizeObjy/2 - S.y)/(E.y-S.y);    
      amin = fmin(a1,a2); amax = fmax(a1,a2);
      bmin = fmin(b1,b2); bmax = fmax(b1,b2);
      lA = fmax(amin, bmin); 
      lB = fmin(amax, bmax);
      A.x = (1-lA) * S.x + lA * E.x;
      B.x = (1-lB) * S.x + lB * E.x;
      A.y = (1-lA) * S.y + lA * E.y;
      B.y = (1-lB) * S.y + lB * E.y;
      intersected = isless(lA, lB); /* lA < lB <=> ray intersects object */
    }

    if (intersected) { 
      double len_AB=sqrt((A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y));

      int N = fmax(1, (int)ceil(len_AB / spObj * SF)); /* Total number of sampling points */
      double *lambda = (double *)malloc(sizeof(double) * N); /* Parameters of sampling points */
      for (int n=0; n<N; n++)
	lambda[n] = lA + (lB-lA)*(1+n)/(N+1);

      double *pixpos = (double *)calloc(pixObjr*pixObjc, sizeof(double)); /* Indicator of contribution pixels */
      /* Remark : in the following Fortran order is used (for Matlab), be careful with the array pixpos */
      for(int n = 1; n < N; n++) { /* Calculate the intersection points */
	T2DVector Q;		/* Sampling point's position */
	Q.x = (1-lambda[n]) * S.x + lambda[n] * E.x;
	Q.y = (1-lambda[n]) * S.y + lambda[n] * E.y;

	T2DPixel qq = pos2pix(Q.x, Q.y, spObj, sizeObjx, sizeObjy); /* Conversion Q into corresponding image pixel qq */
	int r = qq.row; int c = qq.col;
	//printf("pixel : (%d,%d)\n",r, c);

	T2DVector P = pix2pos(r,c,spObj,sizeObjx, sizeObjy); /* Pixel qq's position */

	double ex = 1 - fabs(Q.x - P.x)/spObj; /* Bi-Interpolation coefficient */
	double ey = 1 - fabs(Q.y - P.y)/spObj; /* Bi-Interpolation coefficient */
	int dc = (Q.x>P.x)?1:-1;	/* Incrementation on column */
	int dr = (Q.y>P.y)?-1:1; 

	pixpos[r*pixObjc+c] += ex*ey; /* Sampling point's corresponding pixel index */
	/* C standard array */
	if (c+dc<pixObjc && c+dc>=0)
	  pixpos[r*pixObjc+c+dc] += (1-ex)*ey; /* Sampling point's corresponding pixel index */

	if (r+dr<pixObjr && r+dr>=0)
	  pixpos[(r+dr)*pixObjc+c] += ex*(1-ey); /* Sampling point's corresponding pixel index */

	if (c+dc<pixObjc && c+dc>=0 && r+dr<pixObjr && r+dr>=0)
	  pixpos[(r+dr)*pixObjr+c+dc] += (1-ex)*(1-ey); /* Sampling point's corresponding pixel index */
      }
      for (int pn=0; pn<pixObjr*pixObjc; pn++) {
	double coeff = pixpos[pn] * spObj;
      	if (pixpos[pn]>0){
	  if (forward) /* Forward projection */
	    Y[dpix] += coeff * X[pn];
	  else /* Backward projection */
	    X[pn] += coeff * Y[dpix];
      	}
      }
      free(pixpos);    
    }
  }
}
