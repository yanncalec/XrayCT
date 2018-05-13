#include "T2DProjector.h"

/* On-the-fly projector based on Blob representation and strip integral with specified pre-integrated footprint table */
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
		    double *X, double *Y, int forward)
{
  /* int fanbeam : fanbeam mode */
  /* double pSrc : Source angle position, 0<=pSrc<2PI */
  /* double rSrc : Rotation center to source plan radius */
  /* double rtDet : Rotation of detector plan wrt to its center */
  /* double rDet : Rotation center to detector plan radius */ 
  /* double sizeDet : Detector size */
  /* double spDet : size of detector pixel */
  /* int pixDet : Detector resolution */  

  /* double rblob : Radius of representation blob */
  /* double radius : Radius of standard blob used for evaluation of T */
  /* double *T : Pre-integrated footprint table */
  /* int nT : Resolution of footprint table, i.e. the length of array ftprt */

  /* double *Nodes_x, int nNodes_x : 1D array of node's center's x-axis (flattened array in C order) */
  /* double *Nodes_y, int nNodes_y : 1D array of node's center's x-axis (flattened array in C order) */

  if (nNodes_x != nNodes_y) {
    printf("The X and Y coordinates of nodes have different length! X : %d, Y : %d\n", nNodes_x, nNodes_y);
    exit(EXIT_FAILURE);
  }  
  int nNodes = nNodes_x;   /* int nNodes : number of nodes */

  //  double blob_dil = radius/rblob; /* Dilation factor of blob radius wrt standard blob */
  double Al, Ar;		/* Influenced interval on detector */
  double A0;  /* distance from L to arriving point of P on detector */
  double Tval[pixDet];

  T2DVector S;			/* Source position */
  T2DVector D;			/* Center of detector plan */
  T2DVector P;		    /* Image nodes (pixel or blob) position */
  T2DVector SP;			
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
  /* L.x = -cos(pSrc)*rDet+sin(pSrc)*sizeDet/2; /\* Left border's position *\/ */
  /* L.y = -sin(pSrc)*rDet-cos(pSrc)*sizeDet/2; */

  /* Object pixels iteration */
  for (int j=0; j<nNodes; j++){
    P.x = *(Nodes_x+j);
    P.y = *(Nodes_y+j);
      
    if(fanbeam){		/* Fan beam */
      S.x = cos(pSrc) * rSrc;	/* Source position in fanbeam */
      S.y = sin(pSrc) * rSrc;
      SP = minus(P,S); /* Vector from S to P */

      double sin_theta = radius / norm(SP); /* theta is the angle formed by the tangent vectors and SP, it's included in [0, pi/2] */
      double cos_theta = sqrt(1 - sin_theta*sin_theta);
	
      T2DVector SP1, SP2;	/* Tangent vectors with blob P */
      SP1.x = cos_theta * SP.x - sin_theta * SP.y; /* Rotation of SP by theta */
      SP1.y = sin_theta * SP.x + cos_theta * SP.y;
      SP2.x = cos_theta * SP.x + sin_theta * SP.y; /* Rotation of SP by -theta */
      SP2.y = -sin_theta * SP.x + cos_theta * SP.y;

      /* Arriving positions of SP1 SP2 on detector plan */
      double l1 = -cross(SP1, minus(L,S))/cross(SP1,dU); /* l1 is the distance on detector plan from L to the arriving point of SP1 */
      double l2 = -cross(SP2, minus(L,S))/cross(SP2,dU);
      Al = fmin(l1,l2); 
      Ar = fmax(l1,l2); 
    }
    else { /* Parallel beam */
      A0 = dot(minus(P,L), dU); 
      Al = A0 - radius;		 
      Ar = A0 + radius;
    }

    if (Al<sizeDet && Ar>0){ /* Ray must be on detector area */
      /* Conversion of Al and Ar to detector pixels */
      int dpixL = (int)floor(Al / spDet); /* Detector pixel including Al */
      int dpixR = (int)ceil(Ar / spDet); /* Detector pixel including Ar */
      dpixL = (dpixL<0) ? 0 : dpixL;
      dpixR = (dpixR>pixDet) ? pixDet : dpixR;

      //double *Tval = (double *)malloc((dpixR-dpixL+2)*sizeof(double)); /* Temporary table of look-up value at left border of pixels*/
      for (int dpix = dpixL; dpix <= dpixR; dpix++){
	if (fanbeam) {
	  T2DVector SE = apcb(minus(L,S), dpix*spDet, dU); /* Vector from S to E.  E is the detector pixel dpix's left border. SE = SL + LE */

	  T2DVector PP = apcb(SP, -dot(SP, SE)/dot(SE,SE), SE); /* -1 * (Vector from P to SE, orthogonal to SE) */
	  double vv = (dot(PP,dU)<=0) ? norm(PP) : -norm(PP); /* Signed projection distance from P to SE : 
								 negative at left side of A0, and positive at right side. 
								 this conforms to the SL view point */	    

	  Tval[dpix-dpixL] = lookuptable(T,nT,1/radius,vv); /* Dilation factor is fixed to 1 */
	}
	else{
	  Tval[dpix-dpixL] = lookuptable(T,nT,1/radius,dpix*spDet-A0); 
	}
	
	if (dpix>dpixL) {	/* dpix are borders of pixels */
	    if (forward)
	      Y[dpix-1] += (Tval[dpix-dpixL]-Tval[dpix-dpixL-1]) * X[j];
	    else
	      X[j] += (Tval[dpix-dpixL]-Tval[dpix-dpixL-1]) * Y[dpix-1];
	}
      }
    }
  }
}
