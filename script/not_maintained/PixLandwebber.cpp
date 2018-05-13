#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg &msg);

int main(int argc, char **argv) {


  TCLAP::CmdLine cmd("2D Reconstruction by MSE using Landwebber iteration on pixel image. The optimization problem:\n\
 min_x |Ax-b|^2, x>=0\n\
 is solved.\n\
Example:\n\
PixLandwebber.run DIR --pix 256 --nonneg --tol 1e-3 --iter 500 -vv\n\
solves the problem (+) with precision 1e-3 and 500 iterations, the reconstruction has resolution ~ 256X256.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name",false,"","string", cmd);
  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<int> pixArg("", "pix", "pixel image model dimension (larger side). [256]", false, 256, "int", cmd);

  TCLAP::SwitchArg nonnegSwitch("", "nonneg", "Positivity constraint on. [off]", cmd, false);
  TCLAP::ValueArg<double> tolArg("", "tol", "Iterations stopping tolerance. [1e-5]", false, 1e-5, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of CG iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string pfix = pfixArg.getValue();
  if (pfix == "")
    pfix = getenv ("SYSNAME");

  string cfg_fname = inpath_fname +  "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();
  
  AlgoParms AP;
  AP.pix = pixArg.getValue();
  AP.tol = tolArg.getValue();
  AP.maxIter = IterArg.getValue();
  AP.nonneg = nonnegSwitch.getValue();

  int verbose = verboseSwitch.getValue();
  size_t gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  // Create output path
  char buffer[256];
  sprintf(buffer, "PixLandwebber_pix[%d]_nonneg[%d]_tol[%1.1e]_iter[%d].%s", AP.pix, AP.nonneg, AP.tol, AP.maxIter, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);
  
  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);

  AP.pixSize = conf.sizeObj.maxCoeff() / AP.pix;
  Array2d sz = conf.sizeObj + AP.pixSize/2;
  AP.dimObj = Array2i((int)(sz.x() / conf.sizeObj.maxCoeff() * AP.pix), (int)(sz.y() / conf.sizeObj.maxCoeff() * AP.pix));

  // Init GPU projector
  PixDrvProjector *P = new PixDrvProjector(conf, AP.dimObj, AP.pixSize); // Projector
  
  ArrayXd Xr = ArrayXd::Zero(P->get_dimX());

  clock_t t0 = clock();
  //ConvMsg msg = LinSolver::Solver_normal_CG(*P, Y, Xr, AP.maxIter, AP.tol, NULL, verbose);
  ConvMsg msg = Algo::Landwebber(*P, Y, Xr, AP.nonneg, AP.tol, AP.maxIter, verbose);
  AP.time = (clock() - t0) / (double)CLOCKS_PER_SEC;

  printf("Reconstruction taken %lf seconds\n", AP.time); 

  ImageXXd imr;
  if (conf.phantom != "") {
    ImageXXd im = CmdTools::imread(conf.phantom.data());
    imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), im.rows(), im.cols());
    // double pixSize = fmax(AP.dimObj.x() *1./ im.cols(), AP.dimObj.y() *1./ im.rows());
    // imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), 1., im.rows(), im.cols(), pixSize);
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE = %f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);

    sprintf(buffer, "%s/ghost", outpath_fname.c_str());
    ImageXXd ghost = im - imr;
    printf("Ghost max Coeff. = %f\tnorm = %f\n", ghost.maxCoeff(), ghost.matrix().norm());
    if (verbose > 1)
      CmdTools::imshow(ghost, "Ghost");
    CmdTools::imsave(ghost, buffer);
  }
  else
    imr = Map<ImageXXd>(Xr.data(), AP.dimObj.y(), AP.dimObj.x());

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), buffer);
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);

  // Save reconstructed image
  sprintf(buffer, "%s/recon_resampled", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);

  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, msg);

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;

  if (verbose > 1) {
    CmdTools::imshow(imr, "Reconstructed image");
  }

  return 0;
}

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg &msg)
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
  fout<<endl;

  fout<<"[Algo Parameters]"<<endl;
  fout<<"nonneg="<<AP.nonneg<<endl;
  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;
  fout<<"snr="<<AP.snr<<endl;
  fout<<"corr="<<AP.corr<<endl;
  fout<<"uqi="<<AP.uqi<<endl;
  fout<<"mse="<<AP.mse<<endl;
  fout<<"si="<<AP.si<<endl;
  fout<<"time="<<AP.time<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"niter="<<msg.niter<<endl;
  fout<<"vobj="<<msg.vobj<<endl;
  fout<<"res="<<msg.res<<endl;

  fout.close();
}
