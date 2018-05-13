// TV reconstruction by Conjugated Gradient method and blob representation

#include "CmdTools.hpp"
#include "T2D.hpp"
//#include "BlobImageTools.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg &msg);
//void save_algo_parms_batch_csv(const string &fname, const BlobCG_Parms &AP, const BlobImage &BI);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by Conjugated Gradient method based on blob");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for reconstruction. [hexagonal grid]", cmd, false);
  TCLAP::ValueArg<double> roiArg("", "roi", "factor to diameter of the circle ROI. [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. By default(roi=1, box=1), the square ROI is the smallest square including circle ROI. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to ideal sampling step of reconstruction grid. [2]", false, 2, "double", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<string> blobnameArg("", "blob", "Name of detail blob profile : 'diff', 'mexhat', 'd4gs'. [mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<size_t> nbScaleArg("", "nbScale", "number of scales in multi-blob model. [1]", false, 1, "size_t", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor for mexhat or d4gauss blob. [2.]", false, 2., "double", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-1]", false, 1e-1, "double", cmd);

  // TCLAP::ValueArg<double> idArg("", "id", "Tikhonov regularization for identity constant. [5]", false, 5, "double", cmd);
  // TCLAP::ValueArg<double> lapArg("", "lap", "Tikhonov regularization for laplacian constant. [50]", false, 50, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "CG iterations stopping tolerance. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<size_t> IterArg("", "iter", "maximum number of CG iterations. [100]", false, 100, "size_t", cmd);

  TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
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
  AP.hex = !catSwitch.getValue();
  AP.gtype = (AP.hex)?_Hexagonal_ : _Cartesian_;
  AP.roi = roiArg.getValue();
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();
  AP.fspls = fsplsArg.getValue();

  AP.nbScale = nbScaleArg.getValue(); 
  AP.blobname = blobnameArg.getValue();
  AP.dil =  dilArg.getValue();

  if (AP.nbScale == 1) {
    AP.blobname = "gauss";
    AP.dil = 1;
  }

  AP.cut_off_err = cut_off_errArg.getValue();
  AP.fcut_off_err = fcut_off_errArg.getValue();

  AP.stripint = stripintSwitch.getValue();

  AP.maxIter = IterArg.getValue();
  AP.tol = tolArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  // Create output path
  char buffer[256];
  sprintf(buffer, "BlobCG_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_tol[%1.1e]_iter[%d]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.tol, AP.maxIter, AP.roi, AP.box, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, AP.box, AP.roi);
  
  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);
  AP.Ynrml = (AP.sgNrml) ? Y.abs().mean() : 1.; // normalization factor to compensate the sinogram value impact
  Y /= AP.Ynrml;

  // Pixel phantom image
  ImageXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  if (AP.blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "diff") {
    BI = BlobImageTools::MultiGaussDiffGauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.cut_off_err, AP.fcut_off_err, true);
  }
  else if (AP.blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err, true);
  }
  else if (AP.blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err, true);
  else {
    cerr<<"Unknown blob profile!"<<endl;
    exit(0);
  }
  // Guarantee that the blob image nodes are all inside FOV
  int err = BlobImageTools::BlobImage_FOV_check(*BI, conf.diamFOV);
  if (err>=0) {
    printf("Warning: blob image active nodes exceed FOV(diamFOV=%f)\n", conf.diamFOV);
    printf("Scale %d: diamROI=%f, sizeObj.x=%f, sizeObj.y=%f\n", err, BI->bgrid[err]->diamROI, BI->bgrid[err]->sizeObj.x(), BI->bgrid[err]->sizeObj.y()); 
    //exit(0);
  }

  // Init GPU Blob projector
  BlobProjector *P;
  if (AP.stripint)
    P = new BlobProjector(conf, BI, 2);
  else {
    if (AP.blobname == "diff")
      P = new BlobProjector(conf, BI, 1);
    else
      P = new BlobProjector(conf, BI, 0);
  }

  LinOp *PtP = new AtAOp(P);

  if (verbose > 2) { 
    cout<<*BI<<endl;
    cout<<*P<<endl;
  }

  ArrayXd Xr = ArrayXd::Zero(PtP->get_dimX());

  clock_t t0 = clock();
  ConvMsg msg = LinSolver::Solver_CG(*PtP, P->backward(Y), Xr, AP.maxIter, AP.tol, verbose);
  AP.time = (clock() - t0) / (double)CLOCKS_PER_SEC;

  printf("Reconstruction taken %lf seconds\n", AP.time); 

  AP.res = (P->forward(Xr)-Y).matrix().norm();
  printf("Residual |Ax-Y|=%f, per detector=%f\n", AP.res, AP.res/sqrt(Y.size()));

  // Denormalization of Xr
  Xr *= AP.Ynrml;

  vector<ImageXXd> Imr = BI->blob2multipixel(Xr, conf.dimObj);
  ImageXXd imr = Tools::multi_imsum(Imr);

  if (conf.phantom != "") {
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);

    sprintf(buffer, "%s/ghost", outpath_fname.c_str());
    ImageXXd ghost = im - imr;
    if (verbose > 1)
      CmdTools::imshow(ghost, "Ghost");
    CmdTools::imsave(ghost, buffer);
  }
  
  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);
  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi");
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, *BI,  msg);
  // if (bcsv_fname != "") {
  //   save_algo_parms_batch_csv(bcsv_fname, AP, *BI);
  // }
  // Save reconstructed image(s)
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);
  if (AP.nbScale > 1)
    CmdTools::multi_imsave(Imr, buffer);

  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, *BI, msg);

  // Change the output directory name if SNR is available
  if (conf.phantom != "") {
    sprintf(buffer, "%s.snr[%2.2f]", outpath_fname.c_str(), AP.snr);
    outpath_fname = CmdTools::rename_outpath(outpath_fname, buffer);
  }
  cout<<"Outputs saved in directory "<<outpath_fname<<endl<<endl;

  if (verbose > 1) {
    CmdTools::imshow(imr, "CG Reconstruction");
    if (AP.nbScale > 1)
      CmdTools::multi_imshow(Imr, "CG Reconstruction");
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
  fout<<"fspls="<<AP.fspls<<endl;
  fout<<"cut_off_err="<<AP.cut_off_err<<endl;
  fout<<"fcut_off_err="<<AP.fcut_off_err<<endl;
  fout<<"box="<<AP.box<<endl;
  fout<<"roi="<<AP.roi<<endl;
  fout<<BI<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"stripint="<<AP.stripint<<endl;
  fout<<"Ynrml="<<AP.Ynrml<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
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
  fout<<"niter="<<msg.niter<<endl;
  //fout<<"vobj="<<msg.vobj<<endl;
  fout<<"CG residual="<<msg.res<<endl;
  fout<<"Normalized residual |Ax-b|="<<AP.res<<endl;
  fout<<"residual |Ax-b|="<<AP.res*AP.Ynrml<<endl;

  fout.close();
}
