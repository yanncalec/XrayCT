// L1 reconstruction by L1AL3 method and blob representation

#include "T2D.hpp"
#include "CmdTools.hpp"

#include <tclap/CmdLine.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <ctime>

using namespace Eigen;
using namespace std;

void save_algo_parms(const string &fname, bool hex, size_t nbNode,
		     size_t nbScale, double dil, const string &blobname, 
		     double cut_off_err, double fcut_off_err,
		     double fspls, bool stripint, bool singleprs,
		     bool eq, bool nonneg, double mu, double beta,
		     double tol_Inn, double tol_Out, size_t maxIter, size_t rwIter, 
		     double epsilon, double snr, double corr, double time, const ConvMsg &msg);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by reweighted L1 minimization using L1AL3 method.\nThis program reads a directory containning the sinogram and completed acquisition configuration, and the reconstruction results are saved in a sub-directory of the same directory. Default parameters work well for almost noise free data (eg. SNR > 50db).", ' ', "0.1");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name [auto.]",false,"","string", cmd);
  //TCLAP::ValueArg<string> pf_fnameArg("", "pf","appendix to outpath_fname",false,"","string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for reconstruction. [hexagonal grid]", cmd, false);
  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to ideal sampling step of reconstruction grid. [8]", false, 8, "double", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  TCLAP::SwitchArg singleprsSwitch("", "float", "use single precision floating projector. [false]", cmd, false);

  TCLAP::ValueArg<size_t> nbScaleArg("", "nbScale", "number of scales in multi-blob image model. [2]", false, 2, "size_t", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor for mexhat or d4gauss blob, fixed to 2 for diffgauss blob. [1.5]", false, 1.5, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "Name of detail blob profile : 'diff', 'mexhat', 'd4gs'. [diff]", false, "diff", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-4]", false, 1e-4, "double", cmd);

  TCLAP::SwitchArg nonnegSwitch("", "nonneg", "use positive constraint. [false]", cmd, false);
  TCLAP::ValueArg<size_t> IterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "size_t", cmd);
  TCLAP::ValueArg<double> tolInnArg("", "tolInn", "inner iterations stopping tolerance. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> tolOutArg("", "tolOut", "outer iterations stopping tolerance. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "penalty coefficient for data fidelity. [0.01]", false, 0.01, "double", cmd);
  TCLAP::ValueArg<double> betaArg("", "beta", "penalty coefficient for L1 penalty. [10]", false, 10, "double", cmd);
  TCLAP::SwitchArg eqSwitch("", "eq", "Equality constraint model. [false]", cmd, false);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "relaxation of reweighting, meaningful for rwIter>1. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<size_t> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "size_t", cmd);

  TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string cfg_fname = inpath_fname + cfg_fnameArg.getValue();
  //string pf_fname = pf_fnameArg.getValue();
  string sg_fname = inpath_fname + sg_fnameArg.getValue();
  
  bool hex = !catSwitch.getValue();
  GridType gtype = (hex)?_Hexagonal_ : _Cartesian_;
  double fspls = fsplsArg.getValue();
  size_t nbScale = nbScaleArg.getValue(); 
  string blobname = (nbScale==1) ? "gauss" : blobnameArg.getValue();
  double dil;
  if (nbScale==1)
    dil = 1;
  else if(blobname == "diff")
    dil = 2;
  else 
    dil =  dilArg.getValue();
    
  double cut_off_err = cut_off_errArg.getValue();
  double fcut_off_err = fcut_off_errArg.getValue();

  bool stripint = stripintSwitch.getValue();
  bool singleprs = singleprsSwitch.getValue();

  bool nonneg = nonnegSwitch.getValue();
  size_t maxIter = IterArg.getValue();
  double tol_Inn = tolInnArg.getValue();
  double tol_Out = tolOutArg.getValue();
  double mu = muArg.getValue();
  double beta = betaArg.getValue();
  bool eq = eqSwitch.getValue();
  size_t rwIter = rwIterArg.getValue();
  double epsilon = epsilonArg.getValue();

  int verbose = verboseSwitch.getValue();
  size_t gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  // Create output path
  char fname[256];
  sprintf(fname, "BlobL1AL3_fspls[%2.1f]_nbScale[%ld]_dil[%2.1f]_blob[%s]_stripint[%d]_mu[%1.1e]_beta[%1.1e]_tolInn[%1.1e]_tolOut[%1.1e]_eq[%d]_rwIter[%ld]", fspls, nbScale, dil, blobname.c_str(), stripint, mu, beta, tol_Inn, tol_Out, eq, rwIter);

  string toto;
  if (outpath_fnameArg.getValue() == "")
    toto = inpath_fname+"/"+fname;  
  else
    toto = inpath_fname+"/"+outpath_fnameArg.getValue();  
  string outpath_fname = toto;
  //cout<<outpath_fname<<endl;
  int n=1;
  while (mkdir(outpath_fname.c_str(), 0777) == -1 && n<256) {
    if (errno == EEXIST) {
      cout<<"File exists : "<<outpath_fname<<endl;
      sprintf(fname, "%s_%d", toto.c_str(), n++);
      outpath_fname = fname;
    }
    else exit(0);
  }

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);
  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);
  // Pixel phantom image
  ArrayXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  if (nbScale == 1) {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, cut_off_err, fcut_off_err);
  }
  else {
  // else if (blobname == "diff")
  //   BI = BlobImageTools::SingleDiffGauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, cut_off_err, fcut_off_err);
    if (blobname == "diff")
      BI = BlobImageTools::MultiGaussDiffGauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, cut_off_err, fcut_off_err);
      //BI = BlobImageTools::MultiGaussDiffGauss_v2(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, cut_off_err, fcut_off_err);
    else if (blobname == "mexhat")
      BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, dil, cut_off_err, fcut_off_err);
    else if (blobname == "d4gs")
      BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, dil, cut_off_err, fcut_off_err);
    else {
      cerr<<"Unknown blob profile!"<<endl;
      exit(0);
    }
  }
  

  clock_t tt = clock();
  // Init GPU Blob projector
  BlobProjector P(conf, BI, !singleprs, stripint);
  printf("Construction of blob projector taken %lf seconds\n", (clock() - tt) / (double)CLOCKS_PER_SEC); 

  if (verbose > 1) { 
    cout<<P<<endl;
    cout<<*BI<<endl;
  }

  // Normalization of mu and Y
  double Ynrml = Y.maxCoeff(); // normalization factor to compensate the sinogram value impact
  Y /= Ynrml;
  //mu *= BI->get_nbNode() * 1./pow(conf.pixDet*conf.nbProj_total, 2.);
  mu *= BI->get_nbNode() * 1./ conf.pixDet*conf.nbProj_total;

 // Initialization
  ArrayXd Xr;
  Xr.setZero(P.get_dimX());

  ArrayXd Lambda, Nu, W; 

  W.setOnes(Xr.size());

  clock_t t0[rwIter+1];		// Reconstruction time
  double recsnr=0, reccorr=0;	// SNR and Corr of reconstruction
  vector<ArrayXXd> Imr;		// Reconstructed multiscale image
  ArrayXXd imr;			// Sum of reconstructed image
  ConvMsg msg;

  for(size_t n=0; n<rwIter; n++) {
    if (verbose)
      cout<<"Reweighted L1 minimization : "<<n<<endl;    

    Lambda.setZero(P.get_dimY()); 
    Nu.setZero(P.get_dimX());

    t0[n] = clock();
    msg = SpAlgo::L1AL3(P, W, Y, Xr, Nu, Lambda, eq, nonneg, mu, beta, tol_Inn, tol_Out, maxIter, verbose,
			&CmdTools::imsave, BI, &(conf.dimObj), &outpath_fname);
    t0[n+1] = clock();

    // Save reconstructed coefficients Xr
    sprintf(fname, "%s/xr_%lu.dat", outpath_fname.c_str(), n);
    CmdTools::savearray(Xr, fname);

    Imr = P.blob2multipixel(Xr, conf.dimObj); // reconstruction of current iteration
    imr = Tools::multi_imsum(Imr);
    // SNR and Corr of reconstruction
    if (conf.phantom != "") {
      recsnr = Tools::SNR(imr, im);
      reccorr = Tools::Corr(imr, im);
    }

    sprintf(fname, "%s/recon_%lu", outpath_fname.c_str(), n);
    CmdTools::imsave(imr, fname);
    CmdTools::multi_imsave(Imr, fname);

    if (verbose) {
      printf("Reconstruction taken %lf seconds\n", (t0[n+1]-t0[n])/(double)CLOCKS_PER_SEC); 
      printf("Corr. = %f\t, SNR = %f\n", reccorr, recsnr);
    }

    // Calculate the weight 
    W = Xr.abs();

    for (size_t m=0; m<W.size(); m++)
      W[m] = 1/(W[m] + epsilon);
  }

  double rectime = (t0[rwIter]-t0[0])/(double)CLOCKS_PER_SEC;
  printf("Total reconstruction time : %lf seconds\n", rectime); 

  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi.dat");

  if (verbose) {
    CmdTools::imshow(imr, "Reconstructed blob image");
    CmdTools::multi_imshow(Imr, "Reconstructed blob image");
  }
  
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", hex,  BI->get_nbNode(), 
		  nbScale, dil, blobname,
		  cut_off_err, fcut_off_err,
		  fspls, stripint, singleprs,
		  eq, nonneg, mu, beta, 
		  tol_Inn, tol_Out, maxIter, rwIter,
		  epsilon, recsnr, reccorr, rectime, msg);

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;

  return 0;
}

void save_algo_parms(const string &fname, bool hex, size_t nbNode,
		     size_t nbScale, double dil, const string &blobname, 
		     double cut_off_err, double fcut_off_err,
		     double fspls, bool stripint, bool singleprs,
		     bool eq, bool nonneg, double mu, double beta,
		     double tol_Inn, double tol_Out, size_t maxIter, size_t rwIter, 
		     double epsilon, double snr, double corr, double time, const ConvMsg &msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[BlobImage Parameters]"<<endl;
  fout<<"hex="<<hex<<endl;
  fout<<"nbNode="<<nbNode<<endl;
  fout<<"fspls="<<fspls<<endl;
  fout<<"blobname="<<blobname<<endl;
  fout<<"nbScale="<<nbScale<<endl;
  if (nbScale > 1)
    fout<<"dil="<<dil<<endl;
  fout<<"cut_off_err="<<cut_off_err<<endl;
  fout<<"fcut_off_err="<<fcut_off_err<<endl;
  fout<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"raytracing="<<!stripint<<endl;
  fout<<"float="<<singleprs<<endl;
  fout<<endl;

  fout<<"[BlobL1AL3 Parameters]"<<endl;
  fout<<"eq="<<eq<<endl;
  fout<<"nonneg="<<nonneg<<endl;
  fout<<"mu="<<mu<<endl;
  fout<<"beta="<<beta<<endl;

  fout<<"tol_Inn="<<tol_Inn<<endl;
  fout<<"tol_Out="<<tol_Out<<endl;
  fout<<"maxIter="<<maxIter<<endl;
  fout<<"rwIter="<<rwIter<<endl;

  fout<<"epsilon="<<epsilon<<endl;
  fout<<"snr="<<snr<<endl;
  fout<<"corr="<<corr<<endl;
  fout<<"time="<<time<<endl;

  fout<<"niter="<<msg.niter<<endl;
  fout<<"res="<<msg.res<<endl;
  fout<<"rres="<<msg.rres<<endl;
  fout<<"l1norm="<<msg.norm<<endl;

  fout.close();
}

  // if (verbose) {
  //   CmdTools::imshow(Y.data(), conf.nbProj_total, conf.pixDet, "Sinogram");
  //   //CmdTools::imshow(Y, conf.dimObj.x()*2, conf.dimObj.y()*2, "recon");
  //   // ArrayXXd Bim = B.blob2pixel(U, 0.1, Array2i(512, 512));
  //   // CmdTools::imshow(Bim, "backproj");
  // } 

  // clock_t t1 = clock();
  // //Algo::TVAL3(P, G, Y, Xr, Nu, Lambda, tveq, aniso, nonneg, mu, beta,tol_Inn, tol_Out, maxIter_Inn, maxIter_Out, verbose, 0);
  // SpAlgo::TVAL3(P, *G, W, Y, Xr, Nu, Lambda, tveq, aniso, nonneg, mu, beta,tol_Inn, tol_Out, maxIter, verbose, 0,
  // 		&CmdTools::imsave, &BI, &(conf.dimObj), &outpath_fname);
  // t1 = clock() - t1;
  // double rectime = t1/(double)CLOCKS_PER_SEC;
  // printf("Reconstruction taken %lf seconds\n", rectime); 

  // Interpolation to screen grid
  // SNR and Correlation of reconstruction
  // ArrayXXd imr = BI.blob2pixel(Xr, conf.dimObj);
