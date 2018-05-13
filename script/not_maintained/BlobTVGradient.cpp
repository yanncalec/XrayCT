// TV minimization by gradient method and blob representation

#include "T2D.hpp"
#include "CmdTools.hpp"

#include <tclap/CmdLine.h>
#include <argp.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;
 
void save_algo_parms(const string &fname, bool hex, size_t upSpl, double epsilon, 
		     size_t fspls, double splStep, size_t frds, double radius,
		     bool nonneg, double mu, double tol, size_t maxIter, 
		     double snr, double corr, double time);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by TVGradient method.\nThis program reads a directory containning the sinogram and completed acquisition configuration, and the reconstruction results are saved in a sub-directory of the same directory.", ' ', "0.1");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name",true,"","string", cmd);

  TCLAP::SwitchArg hexSwitch("x", "hex", "use hexagonal grid for reconstruction. [false]", cmd, false);
  TCLAP::ValueArg<double> fsplsArg("s", "fspls", "factor to ideal sampling step of reconstruction grid. [4]", false, 4, "double", cmd);
  TCLAP::ValueArg<double> frdsArg("r", "frds", "factor to ideal radius of grid blob. [4]", false, 4, "double", cmd);
  TCLAP::ValueArg<size_t> upSplArg("u", "upSpl", "Up-sampling factor for interpolation grid. [2]", false, 2, "size_t", cmd);
  TCLAP::ValueArg<double> epsilonArg("e", "epsilon", "Relaxation for TV norm . [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> muArg("m", "mu", "penalty coefficient for data fidelity. [10]", false, 10, "double", cmd);
  TCLAP::SwitchArg nonnegSwitch("n", "nonneg", "use positive constraint. [false]", cmd, false);
  TCLAP::ValueArg<size_t> IterArg("i", "iter", "maximum number of iterations. [100]", false, 100, "size_t", cmd);
  TCLAP::ValueArg<double> tolArg("t", "tol", "iterations stopping tolerance. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  bool hex = hexSwitch.getValue();
  size_t upSpl = upSplArg.getValue();
  double epsilon = epsilonArg.getValue();
  double mu = muArg.getValue();
  bool nonneg = nonnegSwitch.getValue();
  size_t maxIter = IterArg.getValue();
  double tol = tolArg.getValue();
  int verbose = verboseSwitch.getValue();
  size_t gpuid = gpuidArg.getValue();

  size_t fspls = fsplsArg.getValue();
  size_t frds = frdsArg.getValue();

  string inpath_fname = inpath_fnameArg.getValue();
  string cfg_fname = inpath_fname+"/acq.cfg";
  string sg_fname = inpath_fname+"/sino.dat";
  string outpath_fname = inpath_fname+"/"+outpath_fnameArg.getValue();

  // Create output path
  if (mkdir(outpath_fname.c_str(), 0777) == -1) {
    cerr<<"Error in creating path "<<outpath_fname<<" : "<<strerror(errno)<<endl;
    cerr<<"Existing files will be erased."<<endl;
    //exit(-1);
  }
  
  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);
  if (verbose > 1)
    cout<<conf<<endl;

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);

  // Init image model
  TGBlob B;		// Use standard TG parameters
  double nspDet;	// Normalized detector pixel size
  if(conf.fanbeam)
    nspDet = conf.spDet * (conf.diamFOV/2 + conf.rSrc[0]) / (conf.rDet[0] + conf.rSrc[0]);
  else
    nspDet = conf.spDet;
  double radius = B.eval_radius(nspDet) * frds; // Ideal blob radius
  double splStep = B.eval_splStep(nspDet) * fspls; // Ideal sampling step

  string gtype = (hex)?"hexagonal":"cartesian";
  Grid bgrid(conf.sizeObj, splStep, gtype); // Initial blob grid
  Grid igrid(conf.sizeObj, bgrid.vshape*upSpl, splStep/upSpl, gtype); // Interpolation grid
  BlobImage BI(B, bgrid, radius); // Blob-image defined by blob and grid

  // Init GPU Blobdriven projector
  GPULTProjector LT(conf, B.StripTable());
  BlobDrvGPU P(LT, BI);
  // BlobDrvGPU P(new GPULTProjector(conf, B.StripTable()), BI);

  // Gradient operator
  //  BlobGrad G(BI, igrid);
  BlobGradFFT G(BI, igrid);

  // Normalization of mu and Y
  //Y /= Y.maxCoeff();
  //  mu *= igrid.nbNode * 1./pow(conf.pixDet*conf.nbProj_total, 2);
  mu *= igrid.nbNode * 1./ conf.pixDet*conf.nbProj_total;
  ArrayXd Xr = P.backward(Y); // Initialization

  clock_t t1 = clock();
  SpAlgo::TVGradient(P, G, Y, Xr, epsilon, mu, nonneg, tol, maxIter, verbose, 0,
		   &CmdTools::imsave, &BI, &(conf.dimObj), &outpath_fname);
  t1 = clock() - t1;
  double rectime = t1/(double)CLOCKS_PER_SEC;
  printf("TVGradient reconstruction taken %lf seconds\n", rectime); 

  // Interpolation to screen grid
  // SNR and Correlation of reconstruction
  ArrayXXd im = CmdTools::imread(conf.phantom.data());
  ArrayXXd imr = BI.blob2pixel(Xr, conf.dimObj);
  vector<ArrayXXd> Dimr = BI.blob2pixelgrad(Xr, conf.dimObj);
  double recsnr = Tools::SNR(imr, im);
  double reccorr = Tools::Corr(imr, im);
  printf("Corr. = %f\t, SNR = %f\n", reccorr, recsnr);
  
  if (verbose > 1) {
    CmdTools::imshow(imr, "Reconstructed blob image");
    CmdTools::imshow(Dimr[0], "Reconstructed blob image gradX");
    CmdTools::imshow(Dimr[1], "Reconstructed blob image gradY");
    // cout<<"TV norm of pixel image DX = "<<sqrt(Dimr[0]*Dimr[0]).sum()<<endl;
    // cout<<"TV norm of pixel image DY = "<<sqrt(Dimr[1]*Dimr[1]).sum()<<endl;
    // cout<<"TV norm of pixel image = "<<sqrt(Dimr[0]*Dimr[0] + Dimr[1]*Dimr[1]).sum()<<endl;
  }

  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", 
		  hex, upSpl, epsilon,
		  fspls, splStep, frds, radius,
		  nonneg, mu, tol, maxIter,
		  recsnr, reccorr, rectime);

  // Save reconstructed coefficients Xr
  CmdTools::savearray(Xr, outpath_fname+"/xr.dat");

  // Save interpolated blob image and gradient image
  CmdTools::imsave(imr, outpath_fname+"/recon.png");
  CmdTools::imsave(Dimr[0], outpath_fname+"/gradx.png");
  CmdTools::imsave(Dimr[1], outpath_fname+"/grady.png");

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;
  return 0;
}

void save_algo_parms(const string &fname, bool hex, size_t upSpl, double epsilon, 
		     size_t fspls, double splStep, size_t frds, double radius,
		     bool nonneg, double mu, double tol, size_t maxIter,
		     double snr, double corr, double time)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[BlobTVGradient Parameters]"<<endl;
  fout<<"hex="<<hex<<endl;
  fout<<"upSpl="<<upSpl<<endl;
  fout<<"epsilon="<<epsilon<<endl;
  fout<<"fspls="<<fspls<<endl;
  fout<<"frds="<<frds<<endl;
  fout<<"splStep="<<splStep<<endl;
  fout<<"radius="<<radius<<endl;
  fout<<"nonneg="<<nonneg<<endl;
  fout<<"mu="<<mu<<endl;
  fout<<"tol="<<tol<<endl;
  fout<<"maxIter="<<maxIter<<endl;
  fout<<"snr="<<snr<<endl;
  fout<<"corr="<<corr<<endl;
  fout<<"time="<<time<<endl;

  fout.close();
}
