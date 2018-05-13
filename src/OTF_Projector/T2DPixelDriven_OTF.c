#include "T2DProjector.h"

/* 
   Fan-Beam or Parallel-Beam OTF projector  by Pixel-Driven method using an interpolation on detector
*/
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
		     int forward)
{
  double sizeObjx = pixObjc * spObj;
  double sizeObjy = pixObjr * spObj;
  double sizeDet = pixDet * spDet; 

  T2DVector S;	     /* Source position */
  T2DVector D;	     /* Detector center position */
  T2DVector A;	     /* Arriving point's position */
  T2DVector LA;	     /* Vector Detector's upborder - Arriving point */
  T2DVector DA;	     /* Vector Detector's center - Arriving point */

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

  /* /\* Detector up border's position *\/ */
  /* U.x=-cos(pSrc)*rDet+sin(pSrc)*sizeDet/2; */
  /* U.y=-sin(pSrc)*rDet-cos(pSrc)*sizeDet/2; */
	
  /* Object pixels iteration : C order */
  for (int jr=0; jr<pixObjr; jr++) /* Row first */
    for (int jc=0; jc<pixObjc; jc++){
      int pp=jr*pixObjc+jc;     /* Pixel (jr,jc)'s index */

      T2DVector P = pix2pos(jr, jc, spObj, sizeObjx, sizeObjy); /* pixel (jr,jc) position */
      /* P.x=(jc-pixObjr/2+0.5)*spObj; */
      /* P.y=(pixObjc/2-jr-0.5)*spObj; */

      if(fanbeam){ /* Fan beam */
	S.x=cos(pSrc)*rSrc;
	S.y=sin(pSrc)*rSrc;
	A = intersection(L, dU, S, minus(P,S));
	/* double alpha=-(rDet+rSrc)*rSrc/(dot(P,S)-rSrc*rSrc)-1; */
	/* A.x=P.x+alpha*(P.x-S.x); */
	/* A.y=P.y+alpha*(P.y-S.y); */
      }
      else { /* Parallel beam */
	T2DVector LP = minus(P,L);
	A.x = L.x + dot(LP, dU) * dU.x;
	A.y = L.y + dot(LP, dU) * dU.y;
	/* double alpha=1-dot(P,D)/(rDet*rDet); */
	/* A.x=P.x+alpha*D.x; */
	/* A.y=P.y+alpha*D.y; */
      }
      LA = minus(A,L);
      DA = minus(A,D);

      if (norm(DA)<=sizeDet/2) { /* Arrving point is on detector */
	int Bn=floor(norm(LA)/spDet);     /* Arrving point's corresponding detector pixel */
	//double dnc=(2*fabs(Bn-sizeDet/2)-1)*spDet/2; /* center of this bin */
	double das[3];
	das[0]=1-(modulo(norm(LA),spDet))/spDet; /* Distance between A and upborder of this bin */
	das[1]=1;
	das[2]=1-das[0]; /* Distance between A and lowborder of this bin, must satisfy das[0]+das[2]=1 */	
	for(int bin=Bn-1; bin<=Bn+1; bin++) { /* Affected bins */
	  double coeff = das[bin-Bn+1] * spObj * spObj;
	  if (bin<pixDet && bin>=0) {	    
	    if (forward)
	      Y[bin] += coeff * X[pp];
	    else
	      X[pp] += Y[bin] * coeff;
	  }
	}
      }
    }
}
