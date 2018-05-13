#ifndef _BLOBIMAGETOOLS_H_
#define _BLOBIMAGETOOLS_H_

#include "BlobImage.hpp"

//! Useful functions for defining blob image 
namespace BlobImageTools {
  // Evaluate the parameters of Gauss-DiffGauss multi-blob system.
  //  ArrayXd eval_parms_MultiGaussDiffGauss(double W, int nbScale, double cut_off_err=1e-4, double fcut_off_err=1e-2, double fspls=1);
    
  //! Return a single scale Gaussian blob image on a cartesian grid based on a pixel image. This is only for sinogram simulation purpose.
  BlobImage* SingleGaussPix(const Array2i &dimObj, double spObj, double cut_off_err = 1e-4, double fcut_off_err = 1e-2);

  //! Return a single scale Gaussian blob image.
  /*!
    Evaluation of the parameter \f$\alpha\f$ (of grid blob) in function of the detector pixel size.
    The essential bandwidth of image should equal to the essential bandwidth of detector,
    which is of order 1/spDet. By FWHM, the Fourier transform of blob \f$\Phi\f$ evaluated 
    at 1/(2*spDet) should equal to 1/2, therefore the ideal \f$ \alpha = \pi^2\mbox{spDet}^{-2}/(4\log 2) \f$.
    The (truncation) radius is determined through \f$ \Phi(R) = \mbox{cutofferr} \times \Phi(0) \f$, 
    therefore \f$ R  = \sqrt {- \log(\mbox{cutofferr}) /\alpha} \f$.

    Following methods allow to calculate the ideal sampling step \f$h\f$:
    - The FWHM of Gaussblob is \f$ \sqrt{\log 2 / \alpha} \f$. So \f$ h = \sqrt{\log 2 / \alpha} \f$.
    - The standard deviation of Gaussblob is \f$ 1/\sqrt{2 \alpha} \f$, so \f$ h = \sqrt{2 / \alpha} \f$
    - The FWHM of GaussBlob in frequency domain (fFWHM) is \f$ \sqrt{\log(2)\alpha } / \pi \f$. 
    To reduce aliasing on hexagonal grid, \f$h\f$ must be smaller than \f$ 1/\sqrt 3 /\mbox{fFWHM} \f$, so
    \f$ h = \pi/\sqrt{3 \log 2 \alpha} \f$. This method is adapted to hexagonal grid. The value is 1.5 times
    bigger than the method 2.
    - Given the cut_off_err in frequency domain, we get the radius \f$ R_f \f$ in frequency domain :
    \f$ R_f = \sqrt{-\log(\mbox{cutofferr}) \alpha} / \pi \f$. To remove aliasing on hexagonal grid, h must 
    be smaller than \f$ 1/\sqrt 3 / R_f \f$, so \f$ h = \pi/\sqrt{-3\log(\mbox{cutofferr})\alpha} \f$. By taking
    \f$ \mbox{cutofferr} = 0.193 \f$, this gives the same value of h as the method 2.
   
    It's the last mothod which is used in our program. The other multiscale blob system use also this 
    method to determine \f$h\f$. If the reconstruction bandwidth W is divided by a factor 
    \f$C_u\f$, then the \f$h\f$ and \f$R\f$ are modified to \f$h_u = C_u h\f$, and \f$R_u = C_u R\f$.
  */
  BlobImage* SingleGauss(const Array2d &sizeObj, //!< size of square ROI (X, Y) 
			 double diamObj,	 //!< diameter of circular ROI
			 GridType gtype,	 //!< grid type 
			 double W,		 //!< essential bandwidth of image to be represented
			 double cut_off_err=1e-4,	 //!< cut off error in space domain 
			 double fcut_off_err=1e-2	 //!< cut off error in frequency domain
			 );

  //! Return a Multi-scale blob image using Gaussian and DiffGaussian blob.
  /*! 
    It is composed of an Gaussian approximation blob image, and several
    DiffGaussian detail blob image dilated from the same DiffGaussian
    detail blob image using the scaling factor 2.
  */
  BlobImage* MultiGaussDiffGauss(const Array2d &sizeObj, //!< size of square ROI (X, Y) 
				 double diamObj,	 //!< diameter of circular ROI
				 GridType gtype,	 //!< grid type 
				 double W,		 //!< essential bandwidth of image to be represented
				 int nbScale,//!< total number of scales. For example, Gauss + DiffGauss + DiffGauss(2x) is a system of 3 scales.
				 double cut_off_err=1e-4,	 //!< cut off error in space domain 
				 double fcut_off_err=1e-4,	 //!< cut off error in frequency domain
				 bool tightframe=false			 //!< normalization model, true for tightframe, false for l2-normalization
				 );

  //! Return a Multi-scale blob image using Gaussian and Mexican hat blob.
  /*! 
    It is composed of an Gaussian approximation blob image, and several
    Mexican hat detail blob image dilated from the same
    detail blob image using the scaling factor <= 2.
  */
  BlobImage* MultiGaussMexHat(const Array2d &sizeObj, //!< size of square ROI (X, Y) 
			      double diamObj,	 //!< diameter of circular ROI
			      GridType gtype,	 //!< grid type 
			      double W,		 //!< essential bandwidth of image to be represented
			      int nbScale,//!< total number of scales. For example, Gauss + DiffGauss + DiffGauss(2x) is a system of 3 scales.
			      double scaling=1.5,    //!< dilation factor between scale
			      double cut_off_err=1e-4,	 //!< cut off error in space domain 
			      double fcut_off_err=1e-2,	 //!< cut off error in frequency domain
			      bool tightframe=false //!< normalization model, true for tightframe, false for l2-normalization
			      );

  //! Return a Multi-scale blob image using Gaussian and D4Gauss blob.
  /*! 
    It is composed of an Gaussian approximation blob image, and several
    D4Gauss detail blob image dilated from the same
    detail blob image using the scaling factor <= 2.
  */
  BlobImage* MultiGaussD4Gauss(const Array2d &sizeObj, //!< size of square ROI (X, Y) 
			       double diamObj,	 //!< diameter of circular ROI
			       GridType gtype,	 //!< grid type 
			       double W,		 //!< essential bandwidth of image to be represented
			       int nbScale,//!< total number of scales. For example, Gauss + DiffGauss + DiffGauss(2x) is a system of 3 scales.
			       double scaling=1.5,    //!< dilation factor between scale
			       double cut_off_err=1e-4,	 //!< cut off error in space domain 
			       double fcut_off_err=1e-2,	 //!< cut off error in frequency domain
			       bool tightframe=false //!< normalization model, true for tightframe, false for l2-normalization
			       );

  // //! Evaluate the parameters of Gauss Blob image (single blob system)
  // Array3d eval_parms_Gauss(GridType gtype, double W, double cut_off_err=1e-4, double fcut_off_err=1e-2);
  // //! Evaluate the parameters of DiffGauss Blob image (single blob system)
  // Array3d eval_parms_DiffGauss(GridType gtype, double W, double cut_off_err=1e-4, double fcut_off_err=1e-2);
  // //! Evaluate the parameters of Gauss DiffGauss Blob image (multi blob system)
  // ArrayXd eval_parms_MultiGaussDiffGauss(GridType gtype, double W, int nbScale, double cut_off_err=1e-4, double fcut_off_err=1e-2);

  //! Check that a BlobImage object has its grids of different scales bounded inside the FOV of diameter FOV. 

  /*! Non convergence of L1 algorithms has been observed due to the blobs very outside FOV. However
   */
  int BlobImage_FOV_check(const BlobImage &BI, double FOV);

  //! Save a BlobImage object into a file
  void save(const BlobImage &BI, const string &fname) ;
  //! Load a BlobImage object from a file
  BlobImage * load(const string &fname) ;

}
#endif

  // BlobImage* MultiGaussDiffGauss_v1(const Array2d &sizeObj, //!< size of square object (X, Y) 
  // 				    GridType gtype,	 //!< grid type 
  // 				    double W,		 //!< essential bandwidth of image to be represented
  // 				    int nbScale,//!< total number of scales. For example, Gauss + DiffGauss + DiffGauss(2x) is a system of 3 scales.
  // 				    double cut_off_err=1e-4,	 //!< cut off error in space domain 
  // 				    double fcut_off_err=1e-4	 //!< cut off error in frequency domain
  // 				    );

  // BlobImage* MultiGaussDiffGauss_v2(const Array2d &sizeObj, //!< size of square object (X, Y) 
  // 				    GridType gtype,	 //!< grid type 
  // 				    double W,		 //!< essential bandwidth of image to be represented
  // 				    int nbScale,//!< total number of scales. For example, Gauss + DiffGauss + DiffGauss(2x) is a system of 3 scales.
  // 				    double cut_off_err=1e-4,	 //!< cut off error in space domain 
  // 				    double fcut_off_err=1e-4	 //!< cut off error in frequency domain
  // 				    );


  // //! Return a single scale DiffGaussian blob image.
  // /*!
  //   The parameters of DiffGaussBlob is evaluated as if it's a low-pass filter.
  //   Remind that DiffGaussBlob is defined in freq domain as :
  //   \f$ \hat\Psi(w)^2 = \hat\Phi(w/2) - \hat\Phi(w) \f$. The parameter \f$ \alpha \f$ and the sampling step
  //   are determined in a similar way as in the case of Gaussian blob, while the radius is calculated through
  //   a numerical procedure. 
  // */
  // BlobImage* SingleDiffGauss(const Array2d &sizeObj, //!< size of square object (X, Y) 
  // 			     GridType gtype,	 //!< grid type 
  // 			     double W,		 //!< essential bandwidth of image to be represented
  // 			     double cut_off_err=1e-4,	 //!< cut off error in space domain 
  // 			     double fcut_off_err=1e-2	 //!< cut off error in frequency domain
  // 			     );

