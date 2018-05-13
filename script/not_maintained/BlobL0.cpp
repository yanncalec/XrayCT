// Image reconstruction by L1 minimization method and blob representation

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, double fspls, const BlobImage &BI,
		     double cut_off_err, double fcut_off_err, bool stripint, 		     
		     double mu, double tol, int maxIter, int rwIter, double epsilon, 
		     double snr, double corr, double time, const ConvMsg *msg);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by reweighted L0 minimization using IHT based method.", ' ', "0.1");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name [auto.]",false,"","string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for reconstruction. [hexagonal grid]", cmd, false);
  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to ideal sampling step of reconstruction grid. [4]", false, 4, "double", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  TCLAP::SwitchArg notunningSwitch("", "notunning", "no normalization for projector [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-blob image model. [4]", false, 4, "int", cmd);
  //TCLAP::ValueArg<int> modelArg("", "model", "gauss-diffgauss multi-blob image model. [0]", false, 0, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, fixed to 2 for diffgauss blob. [1.5]", false, 1.5, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only for nbScale > 1 : 'diff', 'mexhat', 'd4gs'.[mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. Reduce this value (lower than 1e-4) is not recommended for diff-gauss blob. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-1]", false, 1e-1, "double", cmd);
  TCLAP::SwitchArg tightframeSwitch("", "tf", "use tight frame multiscale system [false]", cmd, false);

  //TCLAP::ValueArg<string> algoArg("", "algo","L0-minimization algorithm : nrml, bb.",false,"bb","string", cmd);

  TCLAP::ValueArg<double> ntermArg("", "nterm", "l0 constraint. The percentage (between 0 and 1) of the biggest coeffcients to be kept. 0 or 1 turns off the constraint.  [0.25]", false, 0.25, "double", cmd);
  TCLAP::ValueArg<int> maxIterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. [1e-3]", false, 1e-3, "double", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();
  
  BlobL0_Parms AP;			// Algo structure
  AP.hex = !catSwitch.getValue();
  GridType gtype = (hex)?_Hexagonal_ : _Cartesian_;
  AP.fspls = fsplsArg.getValue();

  AP.nbScale = nbScaleArg.getValue(); 
  AP.blobname = blobnameArg.getValue();
  AP.dil =  dilArg.getValue();
    
  if (nbScale == 1) {
    blobname = "gauss";
    AP.dil = 1;
  }
  else {
    if (AP.blobname == "diff") 
      AP.dil = 2.0;
  }

  AP.cut_off_err = cut_off_errArg.getValue();
  AP.fcut_off_err = fcut_off_errArg.getValue();
  AP.tightframe = tightframeSwitch.getValue();

  AP.stripint = stripintSwitch.getValue();
  AP.tunning = !notunningSwitch.getValue();
  //string algo = algoArg.getValue();

  AP.nterm_prct = ntermArg.getValue();
  AP.maxIter = maxIterArg.getValue();
  AP.tol = tolArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);
  
  // Create output path
  char buffer[256];
  //sprintf(buffer, "BlobL0_fspls[%2.1f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_stripint[%d]_nterm[%1.1f]_tol[%1.1e]_algo[%s]", fspls, fcut_off_err, nbScale, dil, blobname.c_str(), stripint, nterm_prct, tol, algo.c_str());
  sprintf(buffer, "BlobL0_fspls[%2.1f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_stripint[%d]_nterm[%1.1f]_tol[%1.1e]", fspls, fcut_off_err, nbScale, dil, blobname.c_str(), stripint, nterm_prct, tol);
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);
  // normalize first Y by its maximum coefficient
  double Ynrml = Y.abs().maxCoeff(); // normalization factor to compensate the sinogram value impact
  Y /= Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its maximum coefficient : "<<Ynrml<<endl;

  // Pixel phantom image
  ImageXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  if (blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, cut_off_err, fcut_off_err);
  }
  else if (blobname == "diff") {
    BI = BlobImageTools::MultiGaussDiffGauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, cut_off_err, fcut_off_err, tightframe);
  }
  else if (blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, dil, cut_off_err, fcut_off_err, tightframe);
    //mu *= 0.05;		// between 0.05 and 0.1
  }
  else if (blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, dil, cut_off_err, fcut_off_err, tightframe);
  else {
    cerr<<"Unknown blob profile!"<<endl;
    exit(0);
  }

  clock_t tt = clock();
  // Init GPU Blob projector
  BlobProjector *P;
  double norm_projector;
  DiagOp *M;
  LinOp *A;

  if (stripint)
    P = new BlobProjector(conf, BI, 2);
  else {
    if (blobname == "diff")	// always use table projector for diff-gauss blob
      P = new BlobProjector(conf, BI, 1); 
    else
      P = new BlobProjector(conf, BI, 0);
  }

  if (tunning){			// Normalize the projector
    if (verbose)
      cout<<"Estimation of operator norm..."<<endl;
    norm_projector = P->estimate_opNorm();
    M = new DiagOp(P->get_dimY(), 1./norm_projector);
    A = new CompOp(M, P);
  }
  else
    A = P;

  int nterm = 0;
  if (nterm_prct > 0 and nterm_prct < 1) {
    nterm = (int)ceil(nterm_prct * P->get_dimX());
  }

  if (verbose > 1) { 
    cout<<*P<<endl;
    cout<<*BI<<endl;
  }

 // Initialization
  ArrayXd Xr;// = P->backward(Y);
  Xr.setZero(P->get_dimX());

  ConvMsg msg;
  double recsnr=0, reccorr=0;	// SNR and Corr of reconstruction

  clock_t t0 = clock();
  // if (algo == "nrml")
  //   msg = SpAlgo::L0IHT_Nrml(*P, Y, Xr, nterm, tol, maxIter, verbose);
  // else if (algo == "bb")
  msg = SpAlgo::L0IHT(*A, Y, Xr, nterm, tol, maxIter, verbose);

  double rectime = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", rectime); 

  // Denormalization and descaling Xr
  Xr *= (tunning) ? (Ynrml / norm_projector) : Ynrml;

  vector<ImageXXd> Imr = BI->blob2multipixel(Xr, conf.dimObj);
  ImageXXd imr = Tools::multi_imsum(Imr);
  if (conf.phantom != "") {
    recsnr = Tools::SNR(imr, im);
    reccorr = Tools::Corr(imr, im);
    printf("Corr. = %f\t, SNR = %f\n", reccorr, recsnr);
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);
  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi");
  // // Save reconstruction parameters
  // save_algo_parms(outpath_fname+"/parameters.cfg", fspls, *BI,
  // 		  cut_off_err, fcut_off_err, stripint, 
  // 		  mu, tol, maxIter, rwIter, epsilon,
  // 		  recsnr, reccorr, rectime, msg);

  // Save reconstructed image(s)
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);
  if (nbScale > 1)
    CmdTools::multi_imsave(Imr, buffer);

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;

  if (verbose) {
    CmdTools::imshow(imr, "Reconstructed blob image");
    if (nbScale > 1)
      CmdTools::multi_imshow(Imr, "Reconstructed blob image");
  }

  return 0;
}

