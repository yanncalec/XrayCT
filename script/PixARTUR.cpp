// Image reconstruction by HQ minimization method and pixel representation
#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg msg);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by Edge-Preserving regularization using ARTUR (HQ) method on pixel image. Solve:\n\
 min_x |Ax-b|^2 + mu * sum_i phi(D_ix; alpha) (+)\n	\
Here phi is the edge preserving prior(depending on alpha), and D is the gradient operator.\n \
Example:\n\
PixARTUR.run DIR --pix 256 --prior GM --alpha 1e-3 --mu 1e-4 --tol 1e-3 --iter 50 -vv\n\
solves the problem (+) with GM prior of parameter 1e-3, penalization 1e-4, tolerance 1e-3 and 50 iterations.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::ValueArg<int> pixArg("", "pix", "reconstruction dimension of pixel image (larger side). [256]", false, 256, "int", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. For box=1, the square ROI is the smallest square including circle ROI. [0.7]", false, 0.7, "double", cmd);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<string> priorArg("", "prior", "GM: t*t/(alpha+t*t), nonconvex, approximate l0 norm\nHS: sqrt(alpha+t*t), convex, approximate l1 norm [HS]", false, "HS", "string", cmd);
  TCLAP::ValueArg<double> alphaArg("", "alpha", "Edeg-preserving prior parameter. [1e-3]", false, 1e-3, "double", cmd);

  TCLAP::ValueArg<double> muArg("", "mu", "Edge-preserving prior strength. Recommand values:\nGM prior: ~1e-4 for noiseless data, ~1e-3 for 25db data\nHS prior: ~5e-4 for noiseless data, ~5e-3 for 25db data. [5e-4]", false, 5e-4, "double", cmd);
  TCLAP::ValueArg<double> cgtolArg("", "cgtol", "iterations stopping tolerance for CG. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> cgIterArg("", "cgiter", "maximum number of CG iterations. [200]", false, 200, "int", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "outer iterations stopping tolerance. Decrease it for high precision convergence and noiseless data. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [20]", false, 20, "int", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string pfix = pfixArg.getValue();
  if (pfix == "")
    pfix = getenv ("SYSNAME");

  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();

  AlgoParms AP;			// Algo structure
  //AP.tunning = true;
  AP.pix = pixArg.getValue();
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();

  AP.maxIter = IterArg.getValue();
  AP.tol = tolArg.getValue();
  AP.cgmaxIter = cgIterArg.getValue();
  AP.cgtol = cgtolArg.getValue();
  AP.mu = muArg.getValue();
  AP.alpha = alphaArg.getValue();

  string prior_name = priorArg.getValue();
  if(prior_name=="GM") AP.priortype = _EPP_GM_;
  else if(prior_name=="HS")  AP.priortype = _EPP_HS_;
  // else if(prior_name=="GS")  AP.priortype = _EPP_GS_;
  // else if(prior_name=="GR")  AP.priortype = _EPP_GR_;

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, AP.box, 1.);

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);  
  AP.Ynrml = (AP.sgNrml) ? Y.abs().mean() : 1.; // normalization factor to compensate the sinogram value impact
  Y /= AP.Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its mean abs value: "<<AP.Ynrml<<endl;

  AP.pixSize = conf.sizeObj.maxCoeff() / AP.pix;
  Array2d sz = conf.sizeObj + AP.pixSize/2;
  AP.dimObj = Array2i((int)(sz.x() / conf.sizeObj.maxCoeff() * AP.pix), (int)(sz.y() / conf.sizeObj.maxCoeff() * AP.pix));

  // Projector and Gradient operator
  PixDrvProjector *P = new PixDrvProjector(conf, AP.dimObj, AP.pixSize);
  PixGrad *G = new PixGrad(AP.dimObj, 1);

  // Normalize the projector
  AP.norm_projector = P->estimate_opNorm();
  if (verbose)
    cout<<"Projector is normalized by the estimated norm: "<<AP.norm_projector<<endl;
  
  LinOp *M = new DiagOp(P->get_dimY(), 1./AP.norm_projector);
  LinOp *A = new CompOp(M, P);

  // Initialization
  //ArrayXd Xr = P->backward(Y);
  ArrayXd Xr = ArrayXd::Zero(P->get_dimX());

  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image

  // Create output path
  char buffer[256];
  sprintf(buffer, "PixARTUR_pix[%d]_prior[%s]_mu[%1.1e]_tol[%1.1e].%s", AP.pix, prior_name.c_str(), AP.mu, AP.tol, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  clock_t t0 = clock();
  
  ConvMsg msg = Algo::ARTUR(*A, *G, Y, 
			    Xr, AP.priortype, AP.alpha, AP.mu, 
			    AP.cgtol, AP.cgmaxIter, AP.tol, AP.maxIter, verbose);
  
  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization and descaling Xr
  Xr *= (AP.Ynrml / AP.norm_projector);

  // SNR and Corr of reconstruction
  if (conf.phantom != "") {
    ImageXXd im = CmdTools::imread(conf.phantom.data());
    imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), im.rows(), im.cols(), false);
    // double pixSize = fmax(AP.dimObj.x() *1./ im.cols(), AP.dimObj.y() *1./ im.rows());
    // imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), im.rows(), im.cols(), pixSize);
    // double pixSize1 = fmax(conf.sizeObj.x() / im.cols(), conf.sizeObj.y() / im.rows());
    // imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), AP.pixSize, im.rows(), im.cols(), pixSize1);
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }
  else {
    imr = Map<ImageXXd>(Xr.data(), AP.dimObj.y(), AP.dimObj.x());
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(Xr, AP.dimObj.y(), AP.dimObj.x(), buffer);
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer); // Save also in binary

  // Save reconstructed image
  sprintf(buffer, "%s/recon_resampled", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);

  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, msg);

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;
  
  if (verbose > 1) {
    CmdTools::imshow(imr, "Reconstructed pixel image");
  }

  return 0;
}

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[Pixel image Parameters]"<<endl;
  fout<<"rows="<<AP.dimObj.y()<<endl;
  fout<<"cols="<<AP.dimObj.x()<<endl;
  fout<<"rows*cols="<<AP.dimObj.x()*AP.dimObj.y()<<endl;
  fout<<"size of pixel="<<AP.pixSize<<endl;
  fout<<"box="<<AP.box<<endl;
  fout<<"Ynrml="<<AP.Ynrml<<endl;
  fout<<"Pnorm="<<AP.norm_projector<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"penalization mu="<<AP.mu<<endl;
  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;
  fout<<"prior type="<<AP.priortype<<endl;
  fout<<"EPP parameter alpha="<<AP.alpha<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"snr="<<AP.snr<<endl;
  fout<<"corr="<<AP.corr<<endl;
  fout<<"uqi="<<AP.uqi<<endl;
  fout<<"mse="<<AP.mse<<endl;
  fout<<"si="<<AP.si<<endl;
  fout<<"time="<<AP.time<<endl;

  fout<<"niter="<<msg.niter<<endl;
  fout<<"normalized residual |Ax-b|="<<msg.res<<endl;
  fout<<"EPP norm="<<msg.norm<<endl;

  fout.close();
}

