// Make a C-ARM acquisition configuration 
#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

int main(int argc, char **argv)
{
  string toto;
  bool sgSim = true;
  bool randSrc = false;
  bool fanbeam = true;
  size_t nbProj = 180;
  size_t limView = 1;
  ArrayXd pSrc;
  double pSrc0 = 0;
  double pSrc1 = 0;
  double iSrc = 1;
  double rSrc = 80;
  double rDet = 40;
  //double rtDet = 0;
  size_t pixDet = 512;
  double sizeDet = 50;
  double spDet;
  double diamFOV;

  string fname;
  //  const char *postfix = ".cfg";

  TCLAP::CmdLine cmd("Create 2D C-ARM tomography acquisition configuration file", ' ', "0.1");
  TCLAP::UnlabeledValueArg<string> fnameArg("output_file_name","Output configuration file name",true,"","string");

  TCLAP::SwitchArg fanbeamSwitch("", "fanbeam", "[false]", cmd, false);
  TCLAP::SwitchArg randomSwitch("", "random", "[false]", cmd, false);
  TCLAP::ValueArg<int> nbProjArg("", "nbProj", "[180]", false, 180, "int", cmd);
  TCLAP::ValueArg<double> rSrcArg("", "rSrc", "[80]", false, 80, "double", cmd);
  TCLAP::ValueArg<double> pSrc0Arg("", "pSrc0", "[0]", false, 0, "double", cmd);
  //TCLAP::ValueArg<double> pSrc1Arg("", "pSrc1", "[0]", false, 0, "double", cmd);
  TCLAP::ValueArg<double> iSrcArg("", "iSrc", "[1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> rDetArg("", "rDet", "[40]", false, 40, "double", cmd);
  TCLAP::ValueArg<int> pixDetArg("", "pixDet", "[512]", false, 512, "int", cmd);
  TCLAP::ValueArg<double> sizeDetArg("", "sizeDet", "[50]", false, 50, "double", cmd);

  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.add(fnameArg);
  cmd.parse(argc, argv);

  fname = fnameArg.getValue();
  fanbeam = fanbeamSwitch.getValue();
  randSrc = randomSwitch.getValue();
  nbProj = nbProjArg.getValue();
  rSrc = rSrcArg.getValue();
  pSrc0 = pSrc0Arg.getValue();
  //pSrc1 = pSrc1Arg.getValue();
  iSrc = iSrcArg.getValue();
  rDet = rDetArg.getValue();
  pixDet = pixDetArg.getValue();
  sizeDet = sizeDetArg.getValue();

  pSrc.setZero(nbProj);
  // if (pSrc1>pSrc0)
  //   iSrc = (pSrc1-pSrc0)/(nbProj);

  if (not randSrc)
    for (size_t n = 0; n<nbProj; n++)
      pSrc[n] = (pSrc0 + n*iSrc)/180 * M_PI;  
  else{
    ArrayXd toto = ArrayXd::Random(nbProj);
    toto /= toto.abs().maxCoeff();
    if (fanbeam)
      pSrc = 2*M_PI*toto;
    else
      pSrc = M_PI*toto;
  }

  // Array2d sizeObj = Acq::eval_sizeObj(fanbeam, rSrc, rDet, sizeDet);
  // Array2u dimObj(512,512);
  // if (!sgSim){
  //   printf("Object size in X axis [%f] :", sizeObj.x());
  //   do {
  //     getline(cin, toto);
  //     if (toto != "")
  // 	sizeObj.x() = atof(toto.data());
  //   } while (sizeObj.x() <0);
    
  //   printf("Object size in Y axis [%f] :", sizeObj.y());
  //   do {
  //     getline(cin, toto);
  //     if (toto != "")
  // 	sizeObj.y() = atof(toto.data());
  //   } while (sizeObj.y() <0);    

  //   int pix=512;
  //   printf("Screen display dimension of object in larger axis [%d] :", pix);
  //   do {
  //     getline(cin, toto);
  //     if (toto != "")
  // 	pix = atoi(toto.data());
  //   } while (pix <0);    

  //   if (sizeObj.x()>=sizeObj.y()) {
  //     dimObj.x() = pix;
  //     dimObj.y() = (int)ceil(sizeObj.y() / sizeObj.x() * pix / 2) * 2;
  //   }
  //   else {
  //     dimObj.y() = pix;
  //     dimObj.x() = (int)ceil(sizeObj.x() / sizeObj.y() * pix / 2) * 2;
  //   }
  // }

  // Save a CARM-acquisition configuration to file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[CARM acquisition configuration]"<<endl;
  fout<<"fanbeam="<<fanbeam<<endl;
  fout<<"nbProj="<<nbProj<<endl;
  fout<<"pSrc=";
  for (size_t n = 0; n<nbProj; n++)
    fout<<pSrc[n]<<", ";
  // if (n<nbProj-1) 
  //   fout<<pSrc[n]<<", ";
  // else 
  //   fout<<pSrc[n];
  fout<<endl;
  fout<<"rSrc="<<rSrc<<endl;
  fout<<"rDet="<<rDet<<endl;
  //fout<<"rtDet="<<rtDet<<endl;
  fout<<"sizeDet="<<sizeDet<<endl;
  fout<<"pixDet="<<pixDet<<endl;

  // if (!sgSim) {
  //   fout<<"sizeObj="<<sizeObj.x()<<", "<<sizeObj.y()<<endl;
  //   fout<<"dimObj="<<dimObj.x()<<", "<<dimObj.y()<<endl;
  // }

  fout.close();
  cout<<"Acquisition configuration wrote into file : "<<fname<<endl;
  
  return 0;
}

