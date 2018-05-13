// Image reconstruction by Matching Pursuit and blob representation

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg msg);
void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP, const BlobImage &BI);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by Matching Pursuit.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);
  TCLAP::ValueArg<string> xr_fnameArg("", "xr","reconstructed file name. Reuse the reconstructed coeffcient for reweighted l1 norm",false,"","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for reconstruction. [hexagonal grid]", cmd, false);
  TCLAP::ValueArg<double> roiArg("", "roi", "factor to effective reconstruction ROI. [0.8]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to ideal sampling step of reconstruction grid. [4]", false, 4, "double", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  //TCLAP::SwitchArg notunningSwitch("", "notunning", "no normalization for projector [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-blob image model. [4]", false, 4, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, fixed to 2 for diffgauss blob. [2.0]", false, 2.0, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only for nbScale > 1 : 'diff', 'mexhat', 'd4gs'.[mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. Reduce this value (lower than 1e-4) is not recommended for diff-gauss blob. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-1]", false, 1e-1, "double", cmd);
  TCLAP::SwitchArg frameSwitch("", "frame", "use frame but not tight frame multiscale system [false]", cmd, false);

  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> maxIterArg("", "iter", "maximum number of iterations. [10000]", false, 10000, "int", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  AlgoParms AP;			// Algo structure
  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string bcsv_fname = bcsv_fnameArg.getValue();
  string pfix = pfixArg.getValue();
  if (pfix == "")
    pfix = getenv ("SYSNAME");

  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();
  
  AP.hex = !catSwitch.getValue();
  AP.gtype = (AP.hex)?_Hexagonal_ : _Cartesian_;
  AP.roi = roiArg.getValue();
  AP.fspls = fsplsArg.getValue();

  AP.nbScale = nbScaleArg.getValue(); 
  AP.blobname = blobnameArg.getValue();
  AP.dil =  dilArg.getValue();

  if (AP.nbScale == 1) {
    AP.blobname = "gauss";
    AP.dil = 1;
  }
  else {
    if (AP.blobname == "diff")
      AP.dil = 2.0;
  }

  AP.cut_off_err = cut_off_errArg.getValue();
  AP.fcut_off_err = fcut_off_errArg.getValue();
  AP.tightframe = !frameSwitch.getValue();

  AP.stripint = stripintSwitch.getValue();

  AP.maxIter = maxIterArg.getValue();
  AP.tol = tolArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);
  
  // Create output path
  char buffer[256];
  sprintf(buffer, "BlobMP_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_tol[%1.1e].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.tol, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, true, AP.roi);
  AP.nbProj = conf.nbProj_total;
  AP.pixDet = conf.pixDet;

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);
  double Ynrml = Y.abs().mean(); // normalization factor to compensate the sinogram value impact
  Y /= Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its maximum coefficient : "<<Ynrml<<endl;

  // Pixel phantom image
  ImageXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  if (AP.blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, conf.diamFOV, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "diff") {
    BI = BlobImageTools::MultiGaussDiffGauss(conf.sizeObj, conf.diamFOV, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  }
  else if (AP.blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, conf.diamFOV, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  }
  else if (AP.blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, conf.diamFOV, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  else {
    cerr<<"Unknown blob profile!"<<endl;
    exit(0);
  }

  // Init GPU Blob projector
  BlobProjector *P;

  if (AP.stripint)
    P = new BlobProjector(conf, BI, 2);
  else {
    if (AP.blobname == "diff")	// always use table projector for diff-gauss blob
      P = new BlobProjector(conf, BI, 1); 
    else
      P = new BlobProjector(conf, BI, 0);
  }
  
  if (verbose > 2) { 
    cout<<*BI<<endl;
    cout<<*P<<endl;
  }

  // Initialization
  // ArrayXd W1;
  // for (int lp=0; lp<4; lp++) {
  //   cout<<"lp norm : "<<lp<<endl;
  //   W1 = P->col_lpnorm(lp);
  //   cout<<"Column lp-norm : "<<W1.minCoeff()<<", "<<W1.maxCoeff()<<endl;
  //   W1 = P->row_lpnorm(lp);
  //   cout<<"Row lp-norm : "<<W1.minCoeff()<<", "<<W1.maxCoeff()<<endl;
  // }

  ArrayXd W = P->col_lpnorm(2);

  ArrayXd Xr;
  Xr.setZero(P->get_dimX());

  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image
  clock_t t0 = clock();

  
  ConvMsg msg = SpAlgo::MatchingPursuit(*P, Y, W, Xr, AP.tol, AP.maxIter, verbose);

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  Imr = BI->blob2multipixel(Xr, conf.dimObj);
  imr = Tools::multi_imsum(Imr);
  if (conf.phantom != "") {
    AP.nnz = Tools::l0norm(Xr) *1./ Xr.size();
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("NNZ = %f\tUQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.nnz, AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);
  // if (AP.debias) {
  //   sprintf(buffer, "%s/xr0", outpath_fname.c_str()); // non debiased data
  //   CmdTools::savearray(Xr0, buffer);
  // }

  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi");
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, *BI, msg);
  // if (bcsv_fname != "") {
  //   save_algo_parms_batch_csv(bcsv_fname, AP, *BI);
  // }
  // Save reconstructed image(s)
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);
  if (AP.nbScale > 1)
    CmdTools::multi_imsave(Imr, buffer);

  // Change the output directory name if SNR is available
  if (conf.phantom != "") {
    sprintf(buffer, "%s.snr[%2.2f]", outpath_fname.c_str(), AP.snr);
    outpath_fname = CmdTools::rename_outpath(outpath_fname, buffer);
  }
  cout<<"Outputs saved in directory "<<outpath_fname<<endl;

  if (verbose > 1) {
    CmdTools::imshow(imr, "Reconstructed blob image");
    if (AP.nbScale > 1)
      CmdTools::multi_imshow(Imr, "Reconstructed blob image");
  }

  return 0;
}


void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[BlobImage Parameters]"<<endl;
  fout<<"fspls="<<AP.fspls<<endl;
  fout<<"cut_off_err="<<AP.cut_off_err<<endl;
  fout<<"fcut_off_err="<<AP.fcut_off_err<<endl;
  fout<<BI<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"nnz="<<AP.nnz<<endl;
  fout<<"snr="<<AP.snr<<endl;
  fout<<"corr="<<AP.corr<<endl;
  fout<<"uqi="<<AP.uqi<<endl;
  fout<<"mse="<<AP.mse<<endl;
  fout<<"si="<<AP.si<<endl;
  fout<<"time="<<AP.time<<endl;

  fout<<"niter="<<msg.niter<<endl;
  fout<<"residual |Ax-b|="<<msg.res<<endl;
  fout<<"MP residual="<<msg.norm<<endl;

  fout.close();
}

