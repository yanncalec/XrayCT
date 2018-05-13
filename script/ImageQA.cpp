// Image quality assessment

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("");

  TCLAP::UnlabeledValueArg<string> im_fnameArg("im_fname","Phantom image name",true,"","string", cmd);
  TCLAP::ValueArg<string> imr_fnameArg("", "imr","Reconstructed image name",false,"","string", cmd);
  TCLAP::ValueArg<string> xr_fnameArg("", "xr","Reconstructed data name",false,"","string", cmd);

  TCLAP::ValueArg<double> wlowArg("", "wlow", "[0]", false, 0, "double", cmd);
  TCLAP::ValueArg<double> whighArg("", "whigh", "[2]", false, 2, "double", cmd);

  TCLAP::SwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd, false);

  cmd.parse(argc, argv);

  string im_fname = im_fnameArg.getValue();
  string imr_fname = imr_fnameArg.getValue();
  string xr_fname = xr_fnameArg.getValue();

  double wlow = wlowArg.getValue();
  double whigh = whighArg.getValue();

  bool verbose = verboseSwitch.getValue();

  ImageXXd im = CmdTools::imread(im_fname);
  ImageXXd imr;
  if (xr_fname != "") {
    ArrayXd toto = CmdTools::loadarray(xr_fname, im.size());
    imr = Map<ImageXXd>(toto.data(), im.rows(), im.cols());
  }
  else if (imr_fname != "") 
    imr = CmdTools::imread(imr_fname);
  else {
    cerr<<"Must give one of the arguments : imr_fname, xr_fname"<<endl;
    exit(1);
  }

  bool windowed = false;
  if (wlow > imr.minCoeff()) {	// Apply window
    imr = (imr<wlow).select(0, imr);
    windowed = true;
  }
  if (whigh < imr.maxCoeff()) {
    imr = (imr>whigh).select(whigh, imr);
    windowed = true;
  }
  //CmdTools::imshow(imr, "image");
  if (windowed)
    CmdTools::imsave(imr, "./windowed");

  double recsnr = Tools::SNR(imr, im);
  double reccorr = Tools::Corr(imr, im);
  double recuqi = Tools::UQI(imr, im);
  double recmse = (imr-im).matrix().squaredNorm() / im.size();
  double recsi = Tools::StreakIndex(imr, im);
  printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n\n", recuqi, recmse, reccorr, recsnr, recsi);

  // Save output
  return 0;
}
