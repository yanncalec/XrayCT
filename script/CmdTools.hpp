// Useful tools for command line script

#ifndef _CMDTOOLS_H_
#define _CMDTOOLS_H_

#include <Eigen/Core>		
// Eigen must be included before CImg.h
#include <CImg.h>
#include "T2D.hpp"

using namespace Eigen;
using namespace std;
using namespace cimg_library;

class SimConfig : public AcqConfig {
public :
  Array2d sizeObj;		// shape of rectangular object (HXW)
  double diamROI;		// diameter of ROI
  Array2i dimObj;		// resolution of pixel object (phantom, RXC) for screen display

  double spObj;			// size of phantom's pixel
  string method;		// projection method
  string phantom;		// name of phantom file
  size_t phs;			// source intensity
  double noise_energy;		// energy of noise in sinogram, defined by squareNorm(B-I), with B, I the noisy and noiseless sinogram
  double noise_std;		// standard deviation of noise in sinogram
  double snr;			// snr of sinogram

  SimConfig() : AcqConfig(), sizeObj(), dimObj(), spObj(0), method(""), 
		phantom(""), phs(0), noise_energy(0), noise_std(0), snr(0) 
  {}

  SimConfig(const AcqConfig &conf, 
  	    const Array2d &sizeObj, 
	    double diamROI,
  	    const Array2i &dimObj, 
	    double spObj,
  	    const string &method, 
  	    const string &phantom, 
  	    size_t phs, 
  	    double noise_energy,
  	    double noise_std,
  	    double snr) 
    : AcqConfig(conf), sizeObj(sizeObj), diamROI(diamROI), dimObj(dimObj), spObj(spObj), method(method), 
      phantom(phantom), phs(phs), noise_energy(noise_energy), noise_std(noise_std), snr(snr)
  {}

  SimConfig(bool fanbeam,
  	    int nbProj_total,
  	    const ArrayXd &rSrc,
  	    const ArrayXd &pSrc,
  	    const ArrayXd &rDet,
  	    int pixDet,
  	    double sizeDet,
  	    double diamFOV,
  	    string acq_name,
  	    const Array2d &sizeObj, 
	    double diamROI,
  	    const Array2i &dimObj, // optional, not available for real data
	    double spObj,
  	    const string &method,  // optional
  	    const string &phantom, // optional
  	    size_t phs,  // optional
  	    double noise_energy,
  	    double noise_std,
  	    double snr) //optional
    : AcqConfig(fanbeam, nbProj_total, rSrc, pSrc, rDet, pixDet, sizeDet, diamFOV, acq_name),
      sizeObj(sizeObj), diamROI(diamROI), dimObj(dimObj), spObj(spObj), method(method), 
      phantom(phantom), phs(phs), noise_energy(noise_energy), noise_std(noise_std), snr(snr)
  {}
  friend ostream& operator<<(ostream& out, const SimConfig& conf);
};
  

class AlgoParms {
public:
  // Input Output
  // string inpath_fname;
  // string outpath_fname;
  // string bcsv_fname;
  // string pfix;
  // string cfg_fname;
  // string sg_fname;

  // Acquisition
  int nbProj;
  int pixDet;

  // BlobImage
  bool hex;			// Hexagonal grid
  GridType gtype;		// grid type indicator, same as hex
  double box;			// Square ROI(sizeObj) diameter equals to this value (zoom factor) times the diameter of FOV
  double roi;			// ROI diameter equals to this value (zoom factor) times the diameter of FOV
  double broi;			// Inner square ROI diameter equals to this value (zoom factor) times the diameter of FOV
  double fspls;			// upsampling factor to the image grid and blob radius. It controls indeed the reconstruction bandwidth
  int upSpl;			// upsampling factor to the gradient grid
  bool sgNrml;			// normalize the sinogram
  double Ynrml;			// the normalization value of sinogram
  bool projNrml;		// projector is normalized
  double Pnrml;			// the normalization value of projector

  double cut_off_err;		// space domain cut-off error
  double fcut_off_err;		// frequency domain cut-off error
  int nbScale;			// total number of scales in multiscale blob system
  double dil;			// dilation factor btw scales
  string blobname;		// name of bandpass blob
  bool tightframe;		// follow the tightframe construction for the ms blob system

  // PixelImage
  int pix;			
  Array2i dimObj;
  double pixSize;

  // Projector
  bool stripint;
  bool tunning;

  // // Algo related
  string algo;			// name of algorithm
  string model;			// optimization problem model, EQ, DN, Constraint...

  double reg_Id;		// Tikhonov(GMRF) regularization: PixCG
  double reg_Lap;		// Tikhonov(GMRF) laplacian regularization: PixCG

  double init;			// MART
  double norm_projector;

  bool eq;			// TVAL3, L1PADM
  bool aniso;			// TVAL3
  bool nonneg;			// TVAL3

  bool besov;			// L1, BlobL1
  double besov_p;		// WaveletBesov
  double besov_s;		// WaveletBesov
  int besov_d;			// WaveletBesov
  string wvlname;		// WaveletBesov
  int wvlorder;			// WaveletBesov
  int cgmethod;			// WaveletBesov
  double nz;			// percentage of non zero coeffs desired
  int nnz;			// total number of non zero coeffs
  double sparsity;	    // percentage of non zero coeffs obtained 
  bool debias;

  EPPType priortype;		// Edge-preserving prior(ARTUR, PixHQ)

  double epsilon;		// TVAL3
  double mu;			// TVAL3
  double mu_rel;		// TVAL3
  double alpha;			// PixHQ
  double beta;			// TVAL3, L1ADM
  double beta_rel;		// TVAL3, L1ADM
  double delta;			// L1PADM
  double tau;			// L1, MART
  double gamma;
  bool l1l1;			// PADM
  double nu;			// PADM
  double penalty_grad;
  double penalty_id;
  double tau1;
  double tau2;

  // Convergence precision
  double tol;			// TVAL3
  double stol;			// 
  double tolInn;		// TVAL3
  double tolGap;		// TVAL3
  double cgtol;		// TVAL3
  int maxIter;			// All
  int cgmaxIter;		// TVADM
  double spa;			// sparsity

  // Reweighted iteration
  int rwIter;			// TVAL3, L1PADM
  double rwEpsilon;		// TVAL3, L1PADM

  int hIter;
  double incr;
  bool noapprx;

  // Image quality assessment
  double snr;
  double corr;
  double uqi;
  double mse;
  double si;  
  double time;
  double res;

  // Misc.
  // int verbose;
  // int gpuid;
};


namespace CmdTools {
  string creat_outpath(const string &outpath);
  string rename_outpath(const string &oldname, const string &newname);
  void save_acqcarm(const SimConfig &conf, const string &fname);
 // setting box = false, the square object includes FOV,
 // otherwise it's included in FOV (the inner square). roi is the factor of
 // dilation of reconstruction region
  SimConfig load_acqcarm(const string &fname, double box=1., double roi=1.);
  SimConfig extract_config(const SimConfig &conf, int nbProj, bool endpoint=false);

  ArrayXd loadarray(const string &fname, size_t N);
  ArrayXd loadarray_float(const string &fname, size_t N);
  ArrayXd loadarray(const string &fname, size_t N, bool single_precision, bool endian_swap=false);
  void BenchMark_Op(LinOp &A, int N);
  // void loadarray(const string &fname, ArrayXd &Y);

  template<class T>
  void savearray(const T &X, const string &fname);

  template<class T>
  void multi_savearray(const vector<T> &X, const string &fname);

  template <class T> T endswap(T d);

  ImageXXd imread(const string &fname);
  void imsave(const double *X, int row, int col, const string &fname);
  void imsave(const ArrayXd &X, int row, int col, const string &fname);
  void imsave(const ImageXXd &X, const string &fname);
  void multi_imsave(const vector<ImageXXd> &X, const string &fname);

  void imshow(const double *X, int row, int col, const string &title=0);
  void imshow(const ArrayXd &X, int row, int col, const string &title=0);
  void imshow(const ImageXXd &X, const string& title=0);

  void multi_imshow(const vector<ArrayXd> &X, int row, int col, const string& title=0);
  void multi_imshow(const vector<ImageXXd> &X, const string& title=0);

  void removeSpaces(string &stringIn );
  string extractFilename(string &stringIn);

  // Functions for write and read zip files
  //void writeFileToZipOutputStream( ZipOutputStream &zos, const string &filename );
  //void save_sgsim(const string &fname, const SimConfig &conf, const ArrayXXd &X);
};

#endif

