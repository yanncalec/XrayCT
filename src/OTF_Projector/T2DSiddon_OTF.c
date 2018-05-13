#include "T2DProjector.h"

/* 
Fan-Beam or Parallel-Beam On-The-Fly Projector by Siddon Ray-Driven method 
*/

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
		int forward)
{
  /* int fanbeam : fanbeam mode */
  /* double pSrc : Source angle position, 0<=pSrc<2PI */
  /* double rSrc : Rotation center to source plan radius */
  /* double rDet : Rotation center to detector plan radius */ 
  /* double rtDet : Rotation of detector plan wrt to its center */
  /* double spObj :  side size of object pixel */
  /* int pixObjr, pixObjc : Object's resolution in pixel */ 
  /* double sizeDet : Detector size */
  /* double spDet : size of detector pixel, spDet = sizeDet/pixDet */
  /* int pixDet : Detector resolution */  

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

  /* Detector left-border position, from source's viewpoint */
  
  T2DVector L = apcb(D, -sizeDet/2, dU);

  T2DVector A;  /* In Point, near source */
  T2DVector B;  /* Out Point, near detector */
  int mpix = fmax(pixObjr, pixObjc); /* max of side resolution of image */

  double a1,a2,b1,b2;
  double amin, amax, bmin, bmax;
  double lA, lB;

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

    double len_SE=sqrt((S.x-E.x)*(S.x-E.x)+(S.y-E.y)*(S.y-E.y));

    int intersected;
    /* Calculate the intersection points with object borders : A and B*/
    if (dequal(S.x, E.x)) { /* Source-Detector parallel to OY */
      A.x = (S.x + E.x) / 2;
      B.x = A.x;
      double cc = 2 * (double)(S.y > 0) - 1;
      A.y = cc * sizeObjy / 2;
      B.y = -cc * sizeObjy / 2;
      lA = (A.y - S.y) / (E.y - S.y);
      lB = (B.y - S.y) / (E.y - S.y);
      intersected = isless(fabs(A.x), sizeObjx/2); //(A.x > -sizeObj/2) && (A.x < sizeObj/2);
    }
    else if (dequal(S.y, E.y)){/* Source-Detector parallel to OX */
      A.y = (S.y+E.y)/2; 
      B.y = A.y;
      double cc = 2*(double)(S.x>0)-1;
      A.x = cc * sizeObjx/2;
      B.x = -cc * sizeObjx/2;
      lA = (A.x - S.x) / (E.x - S.x);
      lB = (B.x - S.x) / (E.x - S.x);
      intersected = isless(fabs(A.y), sizeObjy/2); //      intersected = (A.y > -sizeObj/2) && (A.y < sizeObj/2);
    }
    else{
      a1=(sizeObjx/2 - S.x) / (E.x-S.x);
      a2=(-sizeObjx/2 - S.x) / (E.x-S.x);
      b1=(sizeObjy/2 - S.y) / (E.y-S.y);
      b2=(-sizeObjy/2 - S.y) / (E.y-S.y);    
      amin = fmin(a1,a2); amax = fmax(a1,a2);
      bmin = fmin(b1,b2); bmax = fmax(b1,b2);
      lA = fmax(amin, bmin); 
      lB = fmin(amax, bmax);
      A.x = (1-lA) * S.x + lA * E.x;
      B.x = (1-lB) * S.x + lB * E.x;
      A.y = (1-lA) * S.y + lA * E.y;
      B.y = (1-lB) * S.y + lB * E.y;
      intersected = isless(lA, lB);  /* lA < lB <=> ray intersects object */
    }

    if (intersected) {
      double *lambda=(double *)malloc(sizeof(double)*mpix*3);
      int k=1;
      lambda[0] = lA;
      for(int n=-mpix/2; n<=mpix/2; n++){ /* Calculate the lambdas corresponding to intersection */
	if(spObj*n>fmin(A.x,B.x) && spObj*n<fmax(A.x,B.x))
	  lambda[k++]=(spObj*n-S.x)/(E.x-S.x);
	if(spObj*n>fmin(A.y,B.y) && spObj*n<fmax(A.y,B.y))
	  lambda[k++]=(spObj*n-S.y)/(E.y-S.y);
      }
      lambda[k++] = lB;

      /* Rearrange the intersected points by order from source to detector */
      qsort(lambda, k, sizeof(double), qsort_comp_func_ascent);
      
      /* Remove too close points */
      double *L = (double *)malloc(sizeof(double)*(k)); /* L stores the intersection pixel's passby position */
      double *U = (double *)malloc(sizeof(double)*(k)); /* U stores the intersection length */

      int N = 0;      
      for (int n=1; n<k; n++)
	if (isgreater(relative_error(lambda[n], lambda[n-1]), 1e-5)) {
	    L[N] = (lambda[n] + lambda[n-1])/2;
	    U[N++] = lambda[n] - lambda[n-1];
	}
/* 	else */
/* 	  printf("Redundant lambda detected : %d\n",N); */
	  
      L = (double *)realloc(L, sizeof(double)*N);
      U = (double *)realloc(U, sizeof(double)*N);

      /* Get the intersection points cordinates in S->D order */
      int *pixpos = (int *)calloc(pixObjr * pixObjc, sizeof(int)); /* Indicator of processed pixels */

      for(int n = 0; n < N; n++) { /* Calculate the intersection points inside A and B */
	T2DPixel qq = pos2pix((1-L[n]) * S.x + L[n] * E.x, (1-L[n]) * S.y + L[n] * E.y, spObj, sizeObjx, sizeObjy);

	int pn = qq.row * pixObjc + qq.col; /* For C standard array */
	//int pn = qq.col * pixObjr + qq.row;
	if (pixpos[pn]==0) {
	  if (forward){
	    /* Forward projection */
	    Y[dpix] += X[pn] * len_SE * U[n];
	    pixpos[pn] = 1;
	  }
	  else
	    /* Backward projection */
	    X[pn] += Y[dpix] * len_SE * U[n];
	}
	else
	  printf("pixel %d already accessed!\n", pn);
      }
      free(pixpos);
    }
  }
}
