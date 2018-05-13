// Display and save pixel reconstruction results from data files

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("Interpolate and display a region of a pixel image on screen.");

  TCLAP::UnlabeledValueArg<string> im_fnameArg("im_fname_in","Input image name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> imout_fnameArg("im_fname_out","Output image name",true,"","string", cmd);

  TCLAP::ValueArg<string> wvlArg("", "wvl", "wavelet name: haar, db. [db]", false, "db", "string", cmd);
  TCLAP::ValueArg<int> orderArg("", "order", "order of daubechies wavelet: 2, 4...20. [6]", false, 6, "int", cmd);
  TCLAP::ValueArg<double> sparsityArg("", "sp", "sparsity of the desired image [0]", false, 0, "double", cmd);

  //TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
  TCLAP::SwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd, false);

  cmd.parse(argc, argv);

  string im_fname = im_fnameArg.getValue();
  string imout_fname = imout_fnameArg.getValue();
  ImageXXd im = CmdTools::imread(im_fname);

  Array2i dimObj(im.cols(), im.rows());
  string wvl = wvlArg.getValue();
  int order = orderArg.getValue();
  double sp = sparsityArg.getValue();
  bool verbose = verboseSwitch.getValue();

  if (verbose)
    cout<<"Sparsify an image by applying N-term approximation on wavelet coefficients."<<endl;

  GSL_Wavelet2D D(dimObj, wvl, order);
  //cout<<D<<endl;

  Map<ArrayXd> X0(im.data(), im.size());
  //cout<<X0.size()<<endl;
  ArrayXd C0 = D.backward(X0);
  ArrayXd C = SpAlgo::NApprx(C0, (int)ceil(C0.size() * sp));
  ArrayXd X = D.forward(C);
  ImageXXd imr;
  imr = Map<ImageXXd> (X.data(), im.rows(), im.cols());

  double recuqi = Tools::UQI(imr, im);
  double recmse = (imr-im).matrix().squaredNorm() / im.size();
  double recsnr = Tools::SNR(imr, im);
  double reccorr = Tools::Corr(imr, im);
  double recsi = Tools::StreakIndex(imr, im);
  cout<<"Sparsity = "<<sp<<endl;
  cout<<"Sparsified image quality assessment:"<<endl;
  printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", recuqi, recmse, reccorr, recsnr, recsi);

  CmdTools::imsave(imr, imout_fname);
  //CmdTools::savearray(imr, inpath_fname+"/"+rec_fname);

  if (verbose) {
    CmdTools::imshow(imr, "Sparsified image");
  }
  
  cout<<"Outputs saved in file "<<imout_fname<<endl;
  return 0;
}

