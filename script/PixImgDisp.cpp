// Display and save pixel reconstruction results from data files

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("Interpolate and display a region of a pixel image on screen.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> im_fnameArg("im_fname","Phantom image name",false,"","string", cmd);
  //TCLAP::SwitchArg imageSwitch("", "image", "Input coefficient is an image file. [false]", cmd, false);
  TCLAP::ValueArg<int> rowArg("", "row", "Row dimension . [256]", true, 256, "int", cmd);
  TCLAP::ValueArg<int> colArg("", "col", "Column dimension . [256]", true, 256, "int", cmd);

  TCLAP::ValueArg<string> xr_fnameArg("", "xr","Input coefficient file name",false,"xr.dat","string", cmd);
  TCLAP::ValueArg<int> dimArg("", "dim", "Resolution of the larger side of interpolation region . [0]", false, 0, "int", cmd);
  TCLAP::SwitchArg oserrSwitch("", "oserr", "Introduce a offset error in the resampling to prevent perfect fitting. [false]", cmd, false);
  // TCLAP::ValueArg<int> dimrArg("", "dimr", "Row resolution of interpolation region . [0]", false, 0, "int", cmd);
  // TCLAP::ValueArg<int> dimcArg("", "dimc", "Column resolution of interpolation region . [0]", false, 0, "int", cmd);

  //TCLAP::SwitchArg gradSwitch("", "grad", "Gradient of image. [false]", cmd, false);
  TCLAP::ValueArg<double> wlowArg("", "wlow", "[-255]", false, -255, "double", cmd);
  TCLAP::ValueArg<double> whighArg("", "whigh", "[255]", false, 255, "double", cmd);

  //TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
  TCLAP::SwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd, false);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string im_fname = im_fnameArg.getValue();
  string xr_fname = xr_fnameArg.getValue();

  int dim = dimArg.getValue();
  bool oserr = oserrSwitch.getValue();
  // int dimr = dimrArg.getValue();
  // int dimc = dimcArg.getValue();
  int drow = rowArg.getValue();
  int dcol = colArg.getValue();
  int nrow, ncol;		// the dimension to be resampled
  char buffer[256];

  //bool grad = gradSwitch.getValue();
  double wlow = wlowArg.getValue();
  double whigh = whighArg.getValue();

  bool verbose = verboseSwitch.getValue();

  // Load coefficients
  ArrayXd Xr = CmdTools::loadarray(inpath_fname+"/"+xr_fname, drow*dcol);
  if (wlow > Xr.minCoeff())
    Xr = (Xr<wlow).select(0, Xr);
  if (whigh < Xr.maxCoeff())
    Xr = (Xr>whigh).select(whigh, Xr);

  ImageXXd imr = Map<ImageXXd>(Xr.data(), drow, dcol);

  if (im_fname != "") {
    ImageXXd im = CmdTools::imread(im_fname);
    imr = Tools::resampling_piximg(Xr.data(), drow, dcol, im.rows(), im.cols(), oserr);
    // double pixSize = fmax(dcol *1./ im.cols(), drow *1./ im.rows());
    // imr = Tools::resampling_piximg(Xr.data(), drow, dcol, 1., im.rows(), im.cols(), pixSize);
    double recuqi = Tools::UQI(imr, im);
    double recmse = (imr-im).matrix().squaredNorm() / im.size();
    double recsnr = Tools::SNR(imr, im);
    double reccorr = Tools::Corr(imr, im);
    double recsi = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", recuqi, recmse, reccorr, recsnr, recsi);
  }

  if (dim > 0) {
    imr = Tools::resampling_piximg(Xr.data(), Array2i(dcol, drow), dim, oserr);
  }

  // if (wlow > imr.minCoeff())
  //   imr = (imr<wlow).select(0, imr);
  // if (whigh < imr.maxCoeff())
  //   imr = (imr>whigh).select(whigh, imr);

  sprintf(buffer, "Reconstruction");
  string rec_fname = buffer;

  CmdTools::imsave(imr, inpath_fname+"/"+rec_fname);
  //CmdTools::savearray(imr, inpath_fname+"/"+rec_fname);

  if (verbose) {
    CmdTools::imshow(imr, "Reconstruction");
  }
  
  cout<<"Outputs saved in directory "<<inpath_fname<<endl;
  return 0;
}

