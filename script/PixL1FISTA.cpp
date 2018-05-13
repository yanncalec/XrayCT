// The functionality of this file is now integrated into PixL1.cpp, where the projector is normalized.
// This file doesn't normalize the projector, and optimal mu~2.5e-2
// Image reconstruction by Wavelet besov prior minimization method and pixel representation (without tunning)
// Remarks: the FISTA method is simple and efficent. The only important parameter is mu and its default value works quite well
// for the non-normalized projector. The idea of accelerating the convergence by observing the support set doesn't work.

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg msg);
void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP);

int main(int argc, char **argv) {

   TCLAP::CmdLine cmd("2D Reconstruction by L1 minimization on Wavelet representation using FISTA algorithm. \
Solve:\n\
     min |AD*x-b|^2 + mu*|x|_1\n\
A is the X-ray projector, D is the wavelet synthesis operator, and x is the wavelet coefficient.\
Examples:\n\
     PixL1FISTA.run DIR --pix 512 --wvl haar --tol 1e-3 --iter 500 -vv\n\
solves the problem using haar wavelet on a Cartesian grid of dimension 512X512, tolerance 1e-3 and 500 iterations.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::ValueArg<int> BenchArg("", "bench", "Bench mark test (only) for operator. [0: no bench mark test]", false, 0, "int", cmd);
  TCLAP::ValueArg<int> pixArg("", "pix", "reconstruction dimension of pixel image (larger side). [512]", false, 512, "int", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. For box=1, the square ROI is the smallest square including circle ROI. [0.7]", false, 0.7, "double", cmd);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<string> wvlArg("", "wvl", "wavelet name: haar, db. [db]", false, "db", "string", cmd);
  TCLAP::ValueArg<int> orderArg("", "order", "order of daubechies wavelet: 2, 4...20. [6]", false, 6, "int", cmd);

  TCLAP::ValueArg<double> muArg("", "mu", "normalized penalty coefficient for L1 fidelity. The default value works well for data of high SNR (~50 db). [5e-8]", false,  5e-8, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> maxIterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);
  TCLAP::ValueArg<int> FreqArg("", "mfreq", "Message print frequence. [50]", false, 50, "int", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string bcsv_fname = bcsv_fnameArg.getValue();
  string pfix = pfixArg.getValue();
  if (pfix == "")
    pfix = getenv ("SYSNAME");

  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();

  AlgoParms AP;			// Algo structure
  AP.pix = pixArg.getValue();
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();

  AP.wvlname = wvlArg.getValue();
  AP.wvlorder = (AP.wvlname == "haar") ? 2 : orderArg.getValue();

  //AP.algo = algoArg.getValue();
  AP.algo = "fista";
  AP.mu = muArg.getValue();
  AP.maxIter = maxIterArg.getValue();
  AP.tol = tolArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();
  int Freq = FreqArg.getValue();

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
  GSL_Wavelet2D *D = new GSL_Wavelet2D(AP.dimObj, AP.wvlname, AP.wvlorder);
  CompOp *A = new CompOp(P, D);

  // Normalization of mu
  AP.mu_rel = AP.mu;
  {
    double mu_factor = (conf.pixDet * conf.nbProj_total) * log(D->get_dimX()); 
    cout<<"Scaling mu by the factor: "<<mu_factor<<endl;
    AP.mu *= mu_factor;
  }

  if (BenchArg.getValue()>0) {	// Bench-mark test
    int N = BenchArg.getValue();
    cout<<"Pix-driven projector Bench-mark test:"<<endl;
    CmdTools::BenchMark_Op(*P, N);
    cout<<"Wavelet transform Bench-mark test:"<<endl;
    CmdTools::BenchMark_Op(*D, N);
    cout<<"Projector-Wavelet transform Bench-mark test:"<<endl;
    CmdTools::BenchMark_Op(*A, N);
    exit(0);
  }

  // Create output path
  char buffer[256];
  sprintf(buffer, "PixL1FISTA_pix[%d]_wvl[%s]_order[%d]_mu[%1.1e]_tol[%1.1e].%s", AP.pix, AP.wvlname.c_str(), AP.wvlorder, AP.mu_rel, AP.tol, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // Initialization
  ArrayXd Cr = ArrayXd::Zero(D->get_dimX());
  ArrayXd Cr0;
  ImageXXd imr;			// Sum of reconstructed image
  ArrayXd Xr;

  clock_t t0 = clock();
  
  ArrayXd W = ArrayXd::Ones(Cr.size());			       // Reweighting coefficient
  ArrayXd toto(Cr.size());

  ConvMsg msg = SpAlgo::L1FISTA(*A, W, Y, Cr, 
				AP.mu, AP.tol, AP.maxIter, 
				0, 0, verbose, Freq);

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization
  Cr *= AP.Ynrml;
  AP.nnz = Tools::l0norm(Cr);
  AP.sparsity = AP.nnz*1./ Cr.size(); // Be careful, this value is too small due to the zero padding.

  Xr = D->forward(Cr);

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
  if (bcsv_fname != "") {
    save_algo_parms_batch_csv(bcsv_fname, AP);
  }

  // Change the output directory name if SNR is available
  if (conf.phantom != "") {
    sprintf(buffer, "%s.snr[%2.2f]", outpath_fname.c_str(), AP.snr);
    outpath_fname = CmdTools::rename_outpath(outpath_fname, buffer);
  }
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
  fout<<"box="<<AP.box<<endl;
  fout<<"Ynrml="<<AP.Ynrml<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"wvlname="<<AP.wvlname<<endl;
  fout<<"wvlorder="<<AP.wvlorder<<endl;
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
  fout<<"l1 norm="<<msg.norm<<endl;

  fout.close();
}


void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP)
{
  ofstream fp;
  fp.open(fname.data(), ios::app | ios::out);
  if (!fp) {
    cout <<"Cannot open file : "<<fname<<endl;
    exit(1);
  }
  long begin, end;
  fp.seekp(0, ios_base::beg); // put position pointer to the end
  begin = fp.tellp();
  fp.seekp(0, ios_base::end); // put position pointer to the end
  end = fp.tellp();
  char buffer[256];

  if (begin==end) {
    //cout<<"file is empty"<<endl;
    fp<<"PixL1FISTA_batch_reconstruction_results"<<endl;
    // Acquisition
    fp<<"nbProj, "<<"pixDet, ";
    // BlobImage
    fp<<"nbPix, "<<"rows, "<<"cols, "<<"pixSize, ";
    // Algo related
    fp<<"mu_rel, snr, corr, uqi, mse, si, time"<<endl;
  }

  fp<<AP.nbProj<<", "<<AP.pixDet<<", ";
  fp<<AP.dimObj.prod()<<", "<<AP.dimObj.y()<<", "<<AP.dimObj.x()<<", "<<AP.pixSize<<", ";
  fp<<AP.mu_rel<<", "<<AP.snr<<", "<<AP.corr<<", "<<AP.uqi<<", "<<AP.mse<<", "<<AP.si<<", "<<AP.time;
  fp<<endl;

  fp.close();  
}


// Precedent version of mu tunning
    // Explenation :
    // 1/2 |Af-b|^2 + mu*|f|_1
    // The data fitness term |Af-b|^2 ~ M*epsilon, with epsilon the noise level of each detector bin, and M the dimension of data.
    // This is independant of the image model(ie, the underlying basis, the reconstruction dimension etc.)
    // mu must balance the data fitness term and the regularization term.
    // While the regularization term |f|_1 is image model dependant. For the multiresolution wavelet model,
    // the l1 norm of each scale it can be roughly identified (true for l1-normalized wavelet) with the contour length of this scale
    // therefore |f|_1 behaves proportionally to 1/2 * log2(N)-1, with N the number of pixels.
    // The default value mu = 1.5e-6 is calulated from a non-scaled real value (mu=0.025) on the config {nbProj=128, pixDet=512, N=512^2}
    
    // How to balance the data fitness and the regularization term?
    // The model is something like : M * e^2 + mu * S*L, with S the number of scales and L the contour length(constant)
    // Choose mu st M/(S * mu) ~ constant => mu propto M/S
    //double mu_factor = conf.pixDet * conf.nbProj_total / (0.5 * log2(AP.dimObj.prod()) - 1);
    //double mu_factor = (1. * conf.pixDet * conf.nbProj_total) / (D->get_nbScale() - 1.); // This is due to the implementation of GSL
