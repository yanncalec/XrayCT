// Tikhonov regularization based on pixel
#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg &msg);

int main(int argc, char **argv) {
  cout<<"OK 0"<<endl;

  TCLAP::CmdLine cmd("2D Reconstruction by Tikhonov regularization using Conjugated Gradient method on pixel image. Solve:\n\
 min_x |Ax-b|^2 + d1 *|x|^2 + d2 * |Dx|^2 \n\
 Here d1, d2 are respectively the identity and the laplacian regularization constant. D is the gradient operator.\n \
Example:\n\
PixCG.run DIR --pix 256 --id 5 --lap 50 --tol 1e-3 --iter 500 -vv\n\
solves the problem with constant 5, 50, CG precision 1e-3 and 500 iterations, the reconstruction has resolution ~ 256X256.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name",false,"","string", cmd);
  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<int> pixArg("", "pix", "reconstruction dimension of pixel image (larger side). [512]", false, 512, "int", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. For box=1, the square ROI is the smallest square including circle ROI. [0.7]", false, 0.7, "double", cmd);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<double> idArg("", "id", "Tikhonov regularization for identity constant. [5]", false, 5, "double", cmd);
  TCLAP::ValueArg<double> lapArg("", "lap", "Tikhonov regularization for laplacian constant. [50]", false, 50, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "CG iterations stopping tolerance. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of CG iterations. [50]", false, 50, "int", cmd);

  TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();

  string pfix = pfixArg.getValue();
  if (pfix == "")
    pfix = getenv ("SYSNAME");
  cout<<pfix<<endl;

  string cfg_fname = inpath_fname +  "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();
  
  AlgoParms AP;
  AP.pix = pixArg.getValue();
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();

  AP.reg_Id = idArg.getValue();
  AP.reg_Lap = lapArg.getValue();
  AP.tol = tolArg.getValue();
  AP.maxIter = IterArg.getValue();

  int verbose = verboseSwitch.getValue();
  size_t gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  // Create output path
  char buffer[256];
  sprintf(buffer, "PixCG_pix[%d]_regid[%1.1e]_reglap[%1.1e]_tol[%1.1e]_iter[%d].%s", AP.pix, AP.reg_Id, AP.reg_Lap, AP.tol, AP.maxIter, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, AP.box, AP.roi); // if pixel phantom image is given in acq config file, the value 0.95 has no importance.
  
  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);
  AP.Ynrml = (AP.sgNrml) ? Y.abs().mean() : 1.; // normalization factor to compensate the sinogram value impact
  Y /= AP.Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its mean abs value: "<<AP.Ynrml<<endl;

  AP.pixSize = conf.sizeObj.maxCoeff() / AP.pix;
  Array2d sz = conf.sizeObj + AP.pixSize/2;
  double tau = AP.pix / conf.sizeObj.maxCoeff();
  AP.dimObj = Array2i((int)(sz.x() * tau), (int)(sz.y() * tau));

  // Init GPU projector
  PixDrvProjector *P = new PixDrvProjector(conf, AP.dimObj, AP.pixSize); // Projector
  PixGrad *G = new PixGrad(AP.dimObj, 1); // Gradient operator
  IdentOp Id(P->get_dimX()); // Identity operator
  AtAOp PtP(P);
  AtAOp GtG(G);
  //PlusOp L(new PlusOp(&PtP, &GtG, 1, AP.reg_Lap), Id, 1, AP.reg_Id); // Whole system operator
  PlusOp L(&PtP, new PlusOp(&Id, &GtG, AP.reg_Id, AP.reg_Lap), 1, 1); // Whole system operator
  
  ArrayXd Xr = ArrayXd::Zero(P->get_dimX());

  clock_t t0 = clock();
  //ConvMsg msg = LinSolver::Solver_normal_CG(*P, Y, Xr, AP.maxIter, AP.tol, NULL, verbose);
  ConvMsg msg = LinSolver::Solver_CG(L, P->backward(Y), Xr, AP.maxIter, AP.tol, verbose);
  AP.time = (clock() - t0) / (double)CLOCKS_PER_SEC;

  printf("Reconstruction taken %lf seconds\n", AP.time); 

  AP.res = (P->forward(Xr)-Y).matrix().norm();
  printf("Noise (residual) |AX-Y|=%f, |AX-Y|/sqrt(Y.size)=%f\n", AP.res, AP.res/sqrt(Y.size()));

  ArrayXd GX = G->forward(Xr);
  msg.vobj = AP.res*AP.res + AP.reg_Id * Xr.matrix().squaredNorm() + AP.reg_Lap * GX.matrix().squaredNorm();

  // Denormalization of Xr
  Xr *= AP.Ynrml;

  ImageXXd imr;
  if (conf.phantom != "") {
    ImageXXd im = CmdTools::imread(conf.phantom.data());
    imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), im.rows(), im.cols(), false);
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
  fout<<"rows*cols="<<AP.dimObj.x()*AP.dimObj.y()<<endl;
  fout<<"size of pixel="<<AP.pixSize<<endl;
  fout<<"box="<<AP.box<<endl;
  fout<<"Ynrml="<<AP.Ynrml<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"Identity regularization="<<AP.reg_Id<<endl;
  fout<<"Laplacian regularization="<<AP.reg_Lap<<endl;
  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"snr="<<AP.snr<<endl;
  fout<<"corr="<<AP.corr<<endl;
  fout<<"uqi="<<AP.uqi<<endl;
  fout<<"mse="<<AP.mse<<endl;
  fout<<"si="<<AP.si<<endl;
  fout<<"time="<<AP.time<<endl;
  fout<<endl;

  fout<<"niter="<<msg.niter<<endl;
  fout<<"objective function value ="<<msg.vobj<<endl;
  fout<<"CG. residual="<<msg.res<<endl;
  fout<<"normalized residual |Ax-b|="<<AP.res<<endl;
  //fout<<"norm="<<msg.norm<<endl;

  fout.close();
}
