// Display and save Blob reconstruction results from data files

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("Interpolate and display a region of a BlobImage object on screen.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> im_fnameArg("im_fname","Phantom image name",false,"","string", cmd);
  // TCLAP::ValueArg<double> sizeObjxArg("", "sizex", "Size of interpolation region in X axis.", true, 0, "double", cmd);
  // TCLAP::ValueArg<double> sizeObjyArg("", "sizey", "Size of interpolation region in Y axis.", true, 0, "double", cmd);
  TCLAP::ValueArg<int> dimArg("", "dim", "Resolution of the larger side of interpolation region . [512]", false, 512, "int", cmd);
  TCLAP::SwitchArg rawSwitch("", "raw", "Save processed image in binary file format. [false]", cmd, false);

  TCLAP::ValueArg<string> xr_fnameArg("", "xr","Input blob coefficient file name",false,"xr.dat","string", cmd);
  TCLAP::ValueArg<string> bi_fnameArg("", "bi","Input BlobImage object file name",false,"bi.dat","string", cmd);

  TCLAP::SwitchArg prodSwitch("", "prod", "inter-scale product. [false]", cmd, false);
  TCLAP::SwitchArg sumSwitch("", "sum", "Cumulative sum of scales. [false]", cmd, false);
  TCLAP::SwitchArg gradSwitch("", "grad", "Gradient of image. [false]", cmd, false);
  TCLAP::ValueArg<double> spaArg("", "spa", "overall sparsity of the N best term approximation.  [0]", false, 0, "double", cmd);
  TCLAP::ValueArg<double> sclArg("", "scl", "scaling factor btw scales.  [0, use default value of blob image]", false, 0, "double", cmd);
  TCLAP::ValueArg<double> wlowArg("", "wlow", "[-255]", false, -255, "double", cmd);
  TCLAP::ValueArg<double> whighArg("", "whigh", "[255]", false, 255, "double", cmd);

  TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string im_fname = im_fnameArg.getValue();

  // double sizeObjx = sizeObjxArg.getValue();
  // double sizeObjy = sizeObjyArg.getValue();
  int dim = dimArg.getValue();
  bool raw = rawSwitch.getValue();

  string xr_fname = xr_fnameArg.getValue();
  string bi_fname = bi_fnameArg.getValue();

  bool grad = gradSwitch.getValue();
  bool prod = prodSwitch.getValue();
  bool sum = sumSwitch.getValue();
  double spa = spaArg.getValue();
  double scl = sclArg.getValue();
  double wlow = wlowArg.getValue();
  double whigh = whighArg.getValue();

  int verbose = verboseSwitch.getValue();
  size_t gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  // Load BlobImage
  BlobImage *BI = BlobImageTools::load(inpath_fname+"/"+bi_fname);
  // Load blob coefficients
  ArrayXd Xr = CmdTools::loadarray(inpath_fname+"/"+xr_fname, BI->get_nbNode());

  cout<<"Sparsity: "<<endl;
  BI->sparsity(Xr);

  spa = (fabs(spa)>1)? 1 : fabs(spa);

  if (spa > 0) {
    if (prod) {
      ArrayXi PSupp = BI->scalewise_prodmask(Xr, scl, spa);
      Xr *= (PSupp>0).select(1. , ArrayXd::Zero(PSupp.size()));
    }
    else
      Xr = BI->scalewise_NApprx(Xr, scl, spa);
    cout<<"Sparsity after scale-wise thresholding : "<<endl;
    BI->sparsity(Xr);
  }

  Array2d sizeObj = BI->get_sizeObj();
  Array2d toto = sizeObj * dim / sizeObj.maxCoeff();
  Array2i dimObj((int)toto.x(), (int)toto.y()); // image resolution
  char buffer[256];

  if (verbose > 1) {
    cout<<*BI<<endl;
  }

  ImageXXd im;
  if (im_fname != "") {
    im = CmdTools::imread(im_fname);
    dimObj.x() = im.cols(); dimObj.y() = im.rows();
  }

  vector<ImageXXd> Imr = BI->blob2multipixel(Xr, dimObj); // all scales image
  ImageXXd imr = Tools::multi_imsum(Imr);
  if (wlow > imr.minCoeff())
    imr = (imr<wlow).select(wlow, imr);
  if (whigh < imr.maxCoeff())
    imr = (imr>whigh).select(whigh, imr);

  if (im_fname != "") {
    ImageXXd im = CmdTools::imread(im_fname);
    double recuqi = Tools::UQI(imr, im);
    double recmse = (imr-im).matrix().squaredNorm() / im.size();
    double recsnr = Tools::SNR(imr, im);
    double reccorr = Tools::Corr(imr, im);
    double recsi = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", recuqi, recmse, reccorr, recsnr, recsi);
  }

  //sprintf(buffer, "Reconstruction_%d", dim);
  sprintf(buffer, "Reconstruction");
  string rec_fname = buffer;

  CmdTools::imsave(imr, inpath_fname+"/"+rec_fname);
  if (raw)
    CmdTools::savearray(imr, inpath_fname+"/"+rec_fname);

  if (BI->get_nbScale() > 1) {
    CmdTools::multi_imsave(Imr, inpath_fname+"/"+rec_fname);
    if (raw)
      CmdTools::multi_savearray(Imr, inpath_fname+"/"+rec_fname);
  }

  if (verbose) {
    CmdTools::imshow(imr, "Reconstruction");
    if (BI->get_nbScale() > 1)
      CmdTools::multi_imshow(Imr, "Reconstruction");
  }

  if (prod and BI->get_nbScale() > 1 and verbose) {
    vector<ImageXXd> Pr;
    int nbScale = BI->get_nbScale();
    Pr.resize(nbScale-1);
    //ImageXXd Model = Imr[0] / Imr[0].abs().maxCoeff();
    for (int n=0; n<nbScale-1; n++) {
      ImageXXd Model = Imr[n].abs() / Imr[n].abs().maxCoeff();
      Pr[n] = Model * Imr[n+1];
    }
    if (verbose > 1)
      CmdTools::multi_imshow(Pr, "Prod");
    CmdTools::multi_imsave(Pr, inpath_fname+"/Prod");
  }

  if (sum and BI->get_nbScale() > 1 and verbose) {
    vector<ImageXXd> Ims = Tools::multi_imcumsum(Imr); // cumulative summation of scales
    CmdTools::multi_imsave(Ims, inpath_fname+"/"+rec_fname+"_cumsum");
    if (verbose > 1) 
      CmdTools::multi_imshow(Ims, "Reconstruction cumsum");
  }
  
  if (grad) {
    vector<ImageXXd> Dimr = BI->blob2pixelgrad(Xr, dimObj);
    CmdTools::imsave(Dimr[0], inpath_fname+"/gradx");
    CmdTools::imsave(Dimr[1], inpath_fname+"/grady");
  
    if (verbose > 1) {
      CmdTools::imshow(Dimr[0], "Grad in X axis");
      CmdTools::imshow(Dimr[1], "Grad in Y axis");
    }
  }
  
  cout<<"Outputs saved in directory "<<inpath_fname<<endl;
  return 0;
}

