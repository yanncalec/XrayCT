// 2D Acquization configuration class
#ifndef _ACQCONFIG_H_
#define _ACQCONFIG_H_

#include <vector>
#include <sstream>
#include <iostream>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

//#include "T2D.hpp"

//! 2D data acquization configuration class.
/*!  The configuration contains only information concerning
  acquization geometry, but not the model of image nor the projector
  implementation. We suppose that the rotation center is fixed, the
  beam geometry is chosen once for all : fanbeam=true/false, and the
  detector plan is perpendicular to the source-rotation center
  axis. However the source and detector distances to rotation center
  can change at each angular position, which allows to calibrate the
  acquisition system. From the source point of view, the detector bin
  index start by 0 from the left side. Other rules:
  - Detector plan's size is fixed to a maximal virtual size : sizeDet is a scalar. In reality we can concatenate multiple detectors (with some blind area) to enlarge the detector plan.
  - Detector pixel's size (spDet) is fixed, therefore the resolution of detector is also fixed : pixDet is a scalar.
  - Detector's center is always positionned in face of source (pSrc+pi). The source, the object and the center-detector center are colinear.
*/

class AcqConfig {

public :
  bool fanbeam;	      //!< Indicator of fanbeam geometry
  int nbProj_total; //!< Total number of acquisition projections
  ArrayXd rSrc;		//!< Radius of source for each position, i.e. the distance from source to rotation center
  ArrayXd pSrc;		//!< Source angular positions in \f$ [0, 2\pi]\f$
  ArrayXd rDet;	//!< Radius of dectector for each position, i.e. the distance from detector center (the middle) to rotation center
  double spDet;	 //!< Size of detector pixel
  int pixDet; //!< Detector resolution in number of pixels
  double sizeDet;		//!< Length(size) of detector plan.
  double diamFOV;	       //!< Diameter of field of view (FOV)
  string acqname;    //!< Name of the acquizition

public :
  //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  AcqConfig()
    : fanbeam(true), nbProj_total(0), rSrc(), 
      pSrc(), rDet(), spDet(0), pixDet(0), sizeDet(0), 
      diamFOV(0), acqname()
  {};

  AcqConfig(bool, 
	    int, 
	    const ArrayXd&, 
	    const ArrayXd&, 
	    const ArrayXd&, 
	    int, 
	    double, 
	    double,
	    string);

  // AcqConfig(const AcqConfig &);
  // ~AcqConfig();
  friend ostream& operator<<(ostream&, const AcqConfig&);
  friend class Projector;
};


//! Collection of useful functions for acquisition configuration. This
//! module can be extended to more complex acquisition geometry.
namespace Acq{
  //! Create a 2D C-ARM acquisition configuration. 
  /*!
    The 2D C-ARM acquisition is charactized by:
    - rSrc and rDet are constant values
   */
  AcqConfig CARM(bool fanbeam,	//!< Fan-beam geometry
		 const vector<Array2d> &limView, //!< Disjoint view intervals
		 int nbProj_total,		 //!< Total number of acquisition projections
		 double _rSrc,			 //!< Radius of source
		 double _rDet,			 //!< Radius of detector
		 int pixDet,			 //!< Dectector resolution in number of pixels 
		 double sizeDet			 //!< Size of detector 
		 );

  //! Evaluate the diameter of FOV from the acquisition parameters.
  double eval_diamFOV(bool fanbeam, double rSrc, double rDet, double sizeDet);

  //! Evaluate the size of square object (minimal square containing FOV)
  Array2d eval_sizeObj(bool fanbeam, double rSrc, double rDet, double sizeDet);

  //! Given the pixel image shape, estimate the sampling step. Useful for sinogram simulation.
  double eval_splStep(bool fanbeam, double rSrc, double rDet, double sizeDet, const Array2i &shape);

  //! Normalize the size of detector pixel in function of acquisition geometry.
  //! Useful in evaluating the essential bandwidth of reconstruction, which depends on the 
  //! detector resolution.
  double nrml_spDet(const AcqConfig &conf) ;

};

#endif
