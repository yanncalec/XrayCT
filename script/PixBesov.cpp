// Image reconstruction by Wavelet besov prior minimization method and pixel representation 
#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg msg);

int main(int argc, char **argv) {

   TCLAP::CmdLine cmd("2D Reconstruction by Besov norm minimization using nonlinear CG method on Wavelet representation. \
Solve:\n\
     min |A*x-b|^2 + mu{|x|_besov(p,s)}^p\n\
Here p>1 is the smoothness of Besov norm. In order to obtain sparse solution, p value close to 1 is prefered. s>=0 is the order of Besov norm, s=0 seems to give best results.\n\
Examples:\n\
     PixBesov.run DIR --pix 256 --wvl haar --besovp 1.1 --besovs 0 --mu 0.1 --tol 1e-3 --iter 500 -vv\n	\
solves the problem using haar wavelet on a Cartesian grid of dimension 256x256, with Besov norm (1.1, 0), tolerance 1e-3 and 500 iterations.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::ValueArg<int> pixArg("", "pix", "reconstruction dimension of pixel image (larger side). [256]", false, 256, "int", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. For box=1, the square ROI is the smallest square including circle ROI. [0.7]", false, 0.7, "double", cmd);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<string> wvlArg("", "wvl", "wavelet name: haar, db. [db]", false, "db", "string", cmd);
  TCLAP::ValueArg<int> orderArg("", "order", "order of daubechies wavelet: 2, 4...20. [4]", false, 4, "int", cmd);
  TCLAP::ValueArg<double> besovsArg("", "besovs", "Besov norm order [0]", false, 0, "double", cmd);
  TCLAP::ValueArg<double> besovpArg("", "besovp", "Besov norm smoothness [1.1]", false, 1.1, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "penalty coefficient for Besov norm. Suggested values for data of SNR 50db:~2.5e-2 for reconstruction dimension 512X512, ~1e-1 for 256X256. For 25db data: 2.5e-1 for 512X512, 1 for 256X256. [1e-1]", false, 1e-1, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

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
  AP.pix = pixArg.getValue();
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();

  //AP.tunning = false;
  AP.wvlname = wvlArg.getValue();
  AP.besov_p = besovpArg.getValue();
  AP.besov_s = besovsArg.getValue();
  AP.wvlorder = (AP.wvlname == "haar") ? 2 : orderArg.getValue();

  AP.maxIter = IterArg.getValue();
  AP.tol = tolArg.getValue();
  AP.mu = muArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Create output path
  char buffer[256];
  sprintf(buffer, "PixBesov_pix[%d]_wvl[%s]_order[%d]_mu[%1.1e]_besovp[%1.2e]_besovs[%1.2e]_tol[%1.1e].%s", AP.pix, AP.wvlname.c_str(), AP.wvlorder, AP.mu, AP.besov_p, AP.besov_s, AP.tol, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

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

  // Projector and Wavelet operator
  PixDrvProjector *P = new PixDrvProjector(conf, AP.dimObj, AP.pixSize);
  GSL_Wavelet2D *W = new GSL_Wavelet2D(AP.dimObj, AP.wvlname, AP.wvlorder);
  CompOp *A = new CompOp(P, W);

  // Initialization
  ArrayXd Cr = ArrayXd::Zero(W->get_dimX());

  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image

  clock_t t0 = clock();
  
  ConvMsg msg = Algo::WaveletBesov(*A, Y, 
  				   Cr, AP.mu, AP.besov_p, AP.besov_s,
  				   AP.tol, AP.maxIter, verbose);
  
  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization
  Cr *= AP.Ynrml;
  AP.nnz = Tools::l0norm(Cr);
  AP.sparsity = AP.nnz * 1./ Cr.size();
  ArrayXd Xr = W->forward(Cr);

  // SNR and Corr of reconstruction
  if (conf.phantom != "") {
    ImageXXd im = CmdTools::imread(conf.phantom.data());
    imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), im.rows(), im.cols(), false);
    // double pixSize = fmax(AP.dimObj.x() *1./ im.cols(), AP.dimObj.y() *1./ im.rows());
    // imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), im.rows(), im.cols(), pixSize);
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("Sparsity = %f\tUQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.sparsity, AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }
  else {
    imr = Map<ImageXXd>(Xr.data(), AP.dimObj.y(), AP.dimObj.x());
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(Xr, AP.dimObj.y(), AP.dimObj.x(), buffer);
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer); // Save also in binary
  sprintf(buffer, "%s/cr", outpath_fname.c_str());
  CmdTools::savearray(Cr, buffer); // Save wavelet coefficients in binary

  // Save reconstructed image
  sprintf(buffer, "%s/recon_resampled", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);

  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, msg);
  // if (bcsv_fname != "") {
  //   save_algo_parms_batch_csv(bcsv_fname, AP);
  // }

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
  fout<<"pixSize="<<AP.pixSize<<endl;
  //fout<<"sgNrml="<<AP.sgNrml<<endl;
  fout<<"box="<<AP.box<<endl;
  fout<<"Ynrml="<<AP.Ynrml<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"wvlname="<<AP.wvlname<<endl;
  fout<<"wvlorder="<<AP.wvlorder<<endl;
  fout<<"besov_p="<<AP.besov_p<<endl;
  fout<<"besov_s="<<AP.besov_s<<endl;
  fout<<"mu="<<AP.mu<<endl;
  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;
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

  fout<<"niter="<<msg.niter<<endl;
  fout<<"normalized residual |Ax-b|="<<msg.res<<endl;
  fout<<"Besov norm="<<msg.norm<<endl;

  fout.close();
}
