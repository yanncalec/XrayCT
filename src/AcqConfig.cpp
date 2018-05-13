#include "AcqConfig.hpp"

AcqConfig::AcqConfig(bool fanbeam,
		     int nbProj_total,
		     const ArrayXd &rSrc,
		     const ArrayXd &pSrc,
		     const ArrayXd &rDet,
		     int pixDet,
		     double sizeDet,
		     double diamFOV,
		     string acqname)
{
  this->fanbeam = fanbeam;
  this->nbProj_total = nbProj_total;
  
  this->rSrc = rSrc;
  this->pSrc = pSrc;
  this->rDet = rDet;

  this->pixDet = pixDet;
  this->sizeDet = sizeDet;
  this->spDet = sizeDet / pixDet;

  this->diamFOV = diamFOV;	
  this->acqname = acqname;
}

ostream& operator<<(ostream& out, const AcqConfig& conf) 
{
  out<<"----2D Tomography acquization configuration----"<<endl;
  out<<"Configuration : "<<conf.acqname<<endl;
  out<<"Fanbeam : "<<conf.fanbeam<<endl;
  out<<"Total number of projections : "<<conf.nbProj_total<<endl;  
  out<<"Source positions : ";
  for(int n=0; n<conf.nbProj_total; n++)
    out<<conf.pSrc[n]<<", ";
  out<<endl;
  out<<"First source to rotation center distance : "<<conf.rSrc[0]<<endl;
  out<<"First detector to rotation center distance : "<<conf.rDet[0]<<endl;
  out<<"Detector resolution : "<<conf.pixDet<<endl;
  out<<"Detector size : "<<conf.sizeDet<<endl;
  out<<"Diameter of FOV : "<<conf.diamFOV<<endl;
  return out;
}

namespace Acq {

  AcqConfig CARM(bool fanbeam,	//!< Fan-beam geometry
		 const vector<Array2d> &limView, //!< Disjoint view intervals
		 int nbProj_total,		 //!< Total number of acquisition projections
		 double _rSrc,			 //!< Radius of source
		 double _rDet,			 //!< Radius of detector
		 int pixDet,			 //!< Dectector resolution in number of pixels 
		 double sizeDet			 //!< Size of detector 
		 )
  {
    ArrayXd rSrc, rDet, pSrc;
    rSrc.setConstant(nbProj_total, _rSrc);
    rDet.setConstant(nbProj_total, _rDet);
    pSrc.setConstant(nbProj_total, 0);

    double toto = 0;			    // total angle ranges
    int nbView = limView.size();
    ArrayXd agl_ranges(nbView);		    // angle ranges
    ArrayXi prj_pview(nbView); // projections per view interval

    for (int n=0; n<nbView; n++) {
      agl_ranges[n] = limView[n].y() - limView[n].x();
      toto += agl_ranges[n];
    }
    for (int n=0; n<nbView; n++) {
      prj_pview[n] = (int)ceil(agl_ranges[n] / toto * nbProj_total); 
    }  
  
    int pp = 0;
    for (int n=0; n<nbView; n++)
      for (int m=0; m<prj_pview[n]; m++)
	if (pp<nbProj_total)        
	  pSrc[pp++] = limView[n].x() + (agl_ranges[n] / prj_pview[n]) * m; // Angular position of X-Ray source

    double beamDiv, diamFOV;

    if (fanbeam) {
      beamDiv = atan(sizeDet/2/(rDet[0] + rSrc[0])) * 2;
      diamFOV = sin(beamDiv/2) * rSrc[0] * 2; // maximum diameter of object
    }
    else {
      beamDiv = 0;
      diamFOV = sizeDet * 0.99;
    }

    return AcqConfig(fanbeam,
		     nbProj_total,
		     rSrc,
		     pSrc,
		     rDet,
		     pixDet,
		     sizeDet,
		     diamFOV,
		     "C-ARM");
  }
  
  double eval_diamFOV(bool fanbeam, double rSrc, double rDet, double sizeDet)
  {
    double beamDiv, diamFOV;

    if (fanbeam) {
      beamDiv = atan(sizeDet/2/(rDet+rSrc)) * 2;
      diamFOV = sin(beamDiv/2) * rSrc * 2; // maximum diameter of object
    }
    else {
      //beamDiv = 0;
      diamFOV = sizeDet * 0.999;
    }
    return diamFOV;
  }

  Array2d eval_sizeObj(bool fanbeam, double rSrc, double rDet, double sizeDet)
  {
    double diamFOV = eval_diamFOV(fanbeam, rSrc, rDet, sizeDet);
    return Array2d(diamFOV/sqrt(2), diamFOV/sqrt(2));
  }

  double eval_splStep(bool fanbeam, double rSrc, double rDet, double sizeDet, const Array2i &shape)
  {
    return eval_diamFOV(fanbeam, rSrc, rDet, sizeDet) / sqrt(shape[0]*shape[0] + shape[1]*shape[1]);
  }

  double nrml_spDet(const AcqConfig &conf) 
  {
    double nspDet; // Normalized detector pixel size
    if(conf.fanbeam)
      nspDet = conf.spDet * (conf.diamFOV/2. + conf.rSrc[0]) / (conf.rDet[0] + conf.rSrc[0]);
    else
      nspDet = conf.spDet;
    return nspDet;
  }

}
