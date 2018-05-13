// Image reconstruction by TV minimization method and blob representation
// using TVADM algorithm (guaranteed convergence but slow, not recommended)

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg &msg);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by TV minimization on a single scale Gaussian blob image using TVADM algorithm.\n\
This program is identical to BlobTV.run when applying on a single scale blob image (where --nbScale 1 is used), but uses the TVADM algorithm (slower but with theoretically proved convergence) in place of the TVAL3 algorithm.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for blob image and reconstruction. The hexagonal grid requires less samples (86.6%) than cartesian grid for representing a bandlimited function. [hexagonal grid]", cmd, false);

  TCLAP::ValueArg<double> roiArg("", "roi", "ratio of effective reconstruction ROI's diameter to the FOV's diameter. [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. By default(roi=1, box=1), the square ROI is the smallest square including circle ROI. [1]", false, 1, "double", cmd);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to reconstruction bandwidth. [2]", false, 2, "double", cmd);
  TCLAP::ValueArg<int> upSplArg("", "upSpl", "Up-sampling factor for interpolation grid. In theory upSpl>1 increases the presicion of TV norm evaluation (and the computation load), but in practice it has no visible effect for various grid density. [1]", false, 1, "int", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);

  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain, this is a fine-tuning for grid, it increases the grid density without dilating the blob. Decrease this value when the grain or gibbs artifacts are observed. This can seriously increase the computation time. [1e-1]", false, 1e-1, "double", cmd);

  TCLAP::SwitchArg anisoSwitch("", "aniso", "anisotropic TV model. [false]", cmd, false);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "0 for TVEQ, <0 for TV, >0 for TVDN", false, -1, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "penalty coefficient for data fidelity. [1e6]", false, 1e6, "double", cmd);
  TCLAP::ValueArg<double> betaArg("", "beta", "penalty coefficient for gradient penalty. Larger beta accelerate the convergence but reduce the precision. [100]", false, 100, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "outer iterations stopping tolerance. Decrease it for high precision convergence. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [50]", false, 50, "int", cmd);
  TCLAP::ValueArg<double> cgtolArg("", "cgtol", "CG iterations stopping tolerance. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<int> cgIterArg("", "cgiter", "maximum number of CG iterations. [50]", false, 50, "int", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  AlgoParms AP;			// Algo structure

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string pfix = getenv ("SYSNAME");

  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();

  AP.hex = !catSwitch.getValue();
  AP.gtype = (AP.hex)?_Hexagonal_ : _Cartesian_;
  AP.roi = roiArg.getValue();
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();

  AP.fspls = fsplsArg.getValue();
  AP.upSpl = upSplArg.getValue();

  AP.cut_off_err = cut_off_errArg.getValue();
  AP.fcut_off_err = fcut_off_errArg.getValue();

  AP.stripint = stripintSwitch.getValue();

  AP.maxIter = IterArg.getValue();
  AP.tol = tolArg.getValue();
  AP.cgmaxIter = cgIterArg.getValue();
  AP.cgtol = cgtolArg.getValue();

  AP.mu = muArg.getValue();
  AP.beta = betaArg.getValue();
  AP.aniso = anisoSwitch.getValue();
  AP.epsilon = epsilonArg.getValue();
  AP.eq = (AP.epsilon==0);

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Create output path
  char buffer[256];
  sprintf(buffer, "BlobTVADM_fspls[%2.2f]_fcut[%1.1e]_mu[%1.1e]_tol[%1.1e]_epsilon[%1.1e]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err, AP.mu, AP.tol, AP.epsilon, AP.roi, AP.box, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, false, AP.roi);
  AP.nbProj = conf.nbProj_total;
  AP.pixDet = conf.pixDet;

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);  
  AP.Ynrml = (AP.sgNrml) ? Y.abs().mean() : 1.; // normalization factor to compensate the sinogram value impact
  Y /= AP.Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its mean abs value: "<<AP.Ynrml<<endl;

  // Pixel phantom image
  ImageXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  BI = BlobImageTools::SingleGauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
  
  Grid *igrid = new Grid(BI->bgrid[0]->sizeObj, BI->bgrid[0]->vshape*AP.upSpl, BI->bgrid[0]->splStep/AP.upSpl, AP.gtype, conf.diamFOV); // Interpolation grid

  // Init GPU Blob projector
  BlobProjector *P;

  if (AP.stripint)
    P = new BlobProjector(conf, BI, 2);
  else
    P = new BlobProjector(conf, BI, 0);

  // Gradient operator
  BlobInterpl *G = new BlobInterpl(BI, igrid, _Grad_);
  if (verbose > 2) {
    cout<<*P<<endl;
    cout<<*G<<endl;
  }

  // Normalization of mu
  if (AP.epsilon<0) {
    double mu_factor = igrid->nbNode / pow(1.*STD_DIAMFOV, 2.) / (1.* conf.nbProj_total * conf.pixDet);
    cout<<"Scaling mu by the factor: "<<mu_factor<<endl;
    AP.mu_rel = AP.mu;
    AP.mu *= mu_factor;
  }
  else {
    cout<<"Constraint TV minimization model: mu should be set to the same order as beta!"<<endl;
    AP.mu = AP.beta;
  }

  // Initialization
  ArrayXd Xr = ArrayXd::Zero(P->get_dimX());

  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image

  clock_t t0 = clock();

  ConvMsg msg = SpAlgo::TVADM(*P, *G, Y, Xr,
			      AP.epsilon, AP.aniso,
			      AP.mu, AP.beta, 
			      AP.cgtol, AP.cgmaxIter,
			      AP.tol, AP.maxIter, verbose);
  
  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization of Xr
  Xr *= AP.Ynrml;

  Imr = BI->blob2multipixel(Xr, conf.dimObj);
  imr = Tools::multi_imsum(Imr);
  if (conf.phantom != "") {
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);
  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi");
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, *BI,  msg);

  // Save reconstructed image(s)
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);
  // if (AP.nbScale > 1)
  //   CmdTools::multi_imsave(Imr, buffer);

  vector<ImageXXd> Dimr = G->blob2pixelgrad(Xr, conf.dimObj);
  CmdTools::imsave(Dimr[0], outpath_fname+"/gradx");
  CmdTools::imsave(Dimr[1], outpath_fname+"/grady");
  
  cout<<"Outputs saved in directory "<<outpath_fname<<endl;
  
  if (verbose > 1) {
    CmdTools::imshow(imr, "Reconstructed blob image");
    // if (AP.nbScale > 1)
    //   CmdTools::multi_imshow(Imr, "Reconstructed blob image");

    CmdTools::imshow(Dimr[0], "Reconstructed blob image gradX");
    CmdTools::imshow(Dimr[1], "Reconstructed blob image gradY");
  }

  return 0;
}

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg &msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[BlobImage Parameters]"<<endl;
  fout<<"Ratio of squared ROI to squared FOV (no incidence in simulation mode) (box)="<<AP.box<<endl;
  fout<<"Ratio of ROI to FOV (no incidence in simulation mode) (roi)="<<AP.roi<<endl;
  fout<<"fspls="<<AP.fspls<<endl;
  fout<<"cut_off_err="<<AP.cut_off_err<<endl;
  fout<<"fcut_off_err="<<AP.fcut_off_err<<endl;
  fout<<BI<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"stripint="<<AP.stripint<<endl;
  fout<<"Sinogram normalization value (Ynrml)="<<AP.Ynrml<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"upSpl="<<AP.upSpl<<endl;
  //  fout<<"tv2="<<AP.tv2<<endl;
  fout<<"eq="<<AP.eq<<endl;
  fout<<"aniso="<<AP.aniso<<endl;
  fout<<"mu="<<AP.mu<<endl;
  fout<<"beta="<<AP.beta<<endl;

  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;
  fout<<"cgtol="<<AP.cgtol<<endl;
  fout<<"cgmaxIter="<<AP.cgmaxIter<<endl;

  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"snr="<<AP.snr<<endl;
  fout<<"corr="<<AP.corr<<endl;
  fout<<"uqi="<<AP.uqi<<endl;
  fout<<"mse="<<AP.mse<<endl;
  fout<<"si="<<AP.si<<endl;
  fout<<"time="<<AP.time<<endl;

  fout<<"niter="<<msg.niter<<endl;
  fout<<"res="<<msg.res<<endl;
  fout<<"tvnorm="<<msg.norm<<endl;

  fout.close();
}
