// L1-Decoding test

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg *msg);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("L1 decoding test on random generated multiscale blob image.\n\
Solve:\n\
     min |x|_1 st Ax=b (BP)\n\
     min |x|_1 st |Ax-b|^2 < M*epsilon^2 (BPDN)\n\
A is the X-ray projector, and x is the blob coefficient.\
The l1 norm can be reweighted, which is useful in the multistep l1 reconstruction.\n");


  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name [auto.]",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> cfg_fnameArg("acq_config_file", "Acquisition configuration file name",true,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for reconstruction. [hexagonal grid]", cmd, false);
  TCLAP::ValueArg<double> roiArg("", "roi", "factor to effective reconstruction ROI. [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to ideal sampling step of reconstruction grid. [2]", false, 2, "double", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-blob image model. [4]", false, 4, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, fixed to 2 for diffgauss blob. [2.0]", false, 2.0, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only for nbScale > 1 : 'diff', 'mexhat', 'd4gs'.[mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. Reduce this value (lower than 1e-4) is not recommended for diff-gauss blob. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-1]", false, 1e-1, "double", cmd);
  TCLAP::SwitchArg frameSwitch("", "frame", "use frame but not tight frame multiscale system [false]", cmd, false);
  TCLAP::SwitchArg besovSwitch("", "besov", "use Besov norm as initial reweighted l1-norm [false]", cmd, false);

  TCLAP::ValueArg<double> betaArg("", "beta", "PADM : normalized data fitnesss penalty, small beta for large shrinkage effect. [5e3]", false, 5e3, "double", cmd);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "PADM : model selection: 0 for BP, >0 for BPDN with epsilon as noise level per detector, i.e. |Ax-b|^2<M*epsilon^2 with M the sinogram dimension.[0]", false, 0, "double", cmd);
  TCLAP::ValueArg<double> tauArg("", "tau", "PADM : proximal penalty. [1.]", false, 1., "double", cmd);
  TCLAP::ValueArg<double> gammaArg("", "gamma", "PADM : lagrangian step. [1.]", false, 1., "double", cmd);

  TCLAP::ValueArg<double> spaArg("", "spa", "sparsity of random image. [.1]", false, .1, "double", cmd);

  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> maxIterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> rwEpsilonArg("", "rwEpsilon", "reweighting parameter between (0, 1], meaningful for rwIter>1. Blob coefficients bigger than epsilon are treated as support, so small epsilon for strong support detection behavior. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);
  TCLAP::ValueArg<int> FreqArg("", "mfreq", "print convergence message every mfreq iterations. [50]", false, 50, "int", cmd);

  cmd.parse(argc, argv);

  AlgoParms AP;			// Algo structure
  string outpath_fname = outpath_fnameArg.getValue();
  string cfg_fname = cfg_fnameArg.getValue();
  string pfix = pfixArg.getValue();
  if (pfix == "")
    pfix = getenv ("SYSNAME");
  
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
  AP.besov = besovSwitch.getValue();

  AP.stripint = stripintSwitch.getValue();
  AP.algo = "padm"; //algoArg.getValue();

  AP.beta = betaArg.getValue();
  AP.epsilon = epsilonArg.getValue();
  AP.tau = tauArg.getValue();
  AP.gamma = gammaArg.getValue();
  AP.maxIter = maxIterArg.getValue();
  AP.tol = tolArg.getValue();
  AP.spa = spaArg.getValue();

  AP.rwIter = rwIterArg.getValue();
  AP.rwEpsilon = rwEpsilonArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();
  int Freq = FreqArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);
  
  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, true, AP.roi);
  AP.nbProj = conf.nbProj_total;
  AP.pixDet = conf.pixDet;

  BlobImage *BI;
  if (AP.blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "diff") {
    BI = BlobImageTools::MultiGaussDiffGauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  }
  else if (AP.blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  }
  else if (AP.blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
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

  // Generation of random coefficient and the sinogram
  ArrayXd X0=ArrayXd::Random(BI->get_nbNode());
  X0 = SpAlgo::NApprx(X0, (int)ceil(AP.spa*X0.size()));
  //cout<<Tools::l0norm(X0)*1./X0.size()<<endl;
  ArrayXd Noise = ArrayXd::Random(BI->get_nbNode());
  Noise *= AP.epsilon/(Noise).matrix().norm();
  ArrayXd Y = P->forward(X0+Noise);
  
  // Pixel phantom image
  Array2i dimObj(512,512);
  ImageXXd im = BI->blob2pixel(X0, dimObj);
  if (verbose)
    CmdTools::imshow(im, "Phantom");

  // double Ynrml = Y.abs().mean(); // normalization factor to compensate the sinogram value impact
  // Y /= Ynrml;

  // // Normalize the projector
  if (verbose)
    cout<<"Estimation of projector norm..."<<endl;
  AP.norm_projector = P->estimate_opNorm();
  if (verbose)
    cout<<"Projector is normalized by the estimated norm: "<<AP.norm_projector<<endl;
    
  LinOp *M = new DiagOp(P->get_dimY(), 1./AP.norm_projector);
  LinOp *A = new CompOp(M, P);

  if (verbose > 2) { 
    cout<<*BI<<endl;
    cout<<*P<<endl;
  }

  AP.beta_rel = AP.beta;
  double beta_factor = AP.norm_projector / (conf.pixDet * conf.nbProj_total * log(P->get_dimX())); 
  AP.beta *= beta_factor;
  if (verbose)
    cout<<"Scaling beta by the factor: "<<beta_factor<<endl;

  //AP.beta *= 2 * Y.size() / Y.abs().sum();
  //AP.beta *= sqrt(P->get_dimX());

  // Create output path
  char buffer[256];
  sprintf(buffer, "BlobL1Decoding_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_epsilon[%1.1e]_beta[%1.1e]_tol[%1.1e]_rwIter[%d]_rwEpsilon[%2.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.epsilon, AP.beta_rel, AP.tol, AP.rwIter, AP.rwEpsilon, pfix.c_str());
  CmdTools::creat_outpath(outpath_fname + "/" + buffer) ;

  vector<ImageXXd> Imr;		// Reconstructed multiscale image

  ArrayXd Xr;
  ArrayXd Z;			// Lagrangian multiplier
  ArrayXd R;			// Residual or the auxilary variable
  Xr.setZero(X0.size()); 
  Z.setZero(Y.size()); 
  R.setZero(Y.size()); 

  ArrayXd W = ArrayXd::Ones(Xr.size());	// W is the reweighting coefficient
  ArrayXd toto(Xr.size());

  // The following is the reweighted iteration  
  ConvMsg* msg = new ConvMsg[AP.rwIter];
  clock_t t0 = clock();

  for(int n=0; n<AP.rwIter; n++) {
    if (verbose)
      cout<<"\nReweighted L1 minimization iteration : "<<n<<endl;    

    msg[n] = SpAlgo::L1PADM(*A, W, Y, Xr, Z, R,
			    AP.epsilon, 0, AP.beta, AP.tau, AP.gamma, 
			    AP.tol, AP.maxIter, 0, 0, verbose, Freq);

    // update reweighting coefficeint
    toto = Xr.abs();
    double gmax = toto.maxCoeff();
    for (size_t m=0; m<toto.size(); m++)
      toto[m] = 1/(toto[m] + AP.rwEpsilon * gmax);
    W = toto / toto.maxCoeff();

    if (verbose > 1) {
      sprintf(buffer, "%s/rwIter_%d", outpath_fname.c_str(), n);
      ImageXXd imr = BI->blob2pixel(Xr, conf.dimObj);
      CmdTools::imsave(imr, buffer);

      vector<ImageXXd> Imr = BI->blob2multipixel(Xr, conf.dimObj);
      //vector<ImageXXd> Imr = BI->blob2multipixel(W, conf.dimObj);
      CmdTools::multi_imsave(Imr, buffer);
    }
  }

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization of Xr
  //Xr *= (Ynrml / AP.norm_projector);
  Xr /= AP.norm_projector;

  AP.nnz = Tools::l0norm(Xr);
  AP.sparsity = AP.nnz *1./ Xr.size();

  // if (conf.phantom != "") {
  AP.snr = Tools::SNR(Xr, X0);
  AP.corr = Tools::Corr(Xr, X0);
  AP.mse = (Xr-X0).matrix().squaredNorm() / X0.size();
  printf("Sparsity = %f\tMSE=%f\tCorr. = %f\tSNR = %f\n", AP.sparsity, AP.mse, AP.corr, AP.snr);
  // }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);
  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi");
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, *BI, msg);

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;

  return 0;
}

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg *msg)
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

  fout<<"[Projector Parameters]"<<endl;
  fout<<"stripint="<<AP.stripint<<endl;
  fout<<"norm_projector="<<AP.norm_projector<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"algo="<<AP.algo<<endl;
  fout<<"besov="<<AP.besov<<endl;
  if(AP.algo=="padm") {
    fout<<"beta_rel="<<AP.beta_rel<<endl;
    fout<<"beta="<<AP.beta<<endl;
    fout<<"epsilon="<<AP.epsilon<<endl;
    fout<<"gamma="<<AP.gamma<<endl;
    fout<<"tau="<<AP.tau<<endl;
  }
  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;

  fout<<"rwIter="<<AP.rwIter<<endl;
  fout<<"rwEpsilon="<<AP.rwEpsilon<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"nnz="<<AP.nnz<<endl;
  fout<<"sparsity="<<AP.sparsity<<endl;

  fout<<"snr="<<AP.snr<<endl;
  fout<<"corr="<<AP.corr<<endl;
  fout<<"uqi="<<AP.uqi<<endl;
  fout<<"mse="<<AP.mse<<endl;
  fout<<"si="<<AP.si<<endl;
  fout<<"time="<<AP.time<<endl;

  for (int n=0; n<AP.rwIter; n++) {
    fout<<endl<<"Reweighted iteration : "<<n<<endl;
    fout<<"niter="<<msg[n].niter<<endl;
    fout<<"residual |Ax-b|="<<msg[n].res<<endl;
    fout<<"l1 norm="<<msg[n].norm<<endl;
  }

  fout.close();
}

