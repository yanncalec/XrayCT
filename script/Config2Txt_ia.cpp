// Make a C-ARM acquisition configuration (interactive)
#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

int main(int argc, char **argv)
{
  string toto;
  bool sgSim = true;
  bool eqSrc = true;
  bool fanbeam = true;
  size_t nbProj = 180;
  size_t limView = 1;
  double *pSrc;
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
  cmd.add(fnameArg);
  cmd.parse(argc, argv);

  fname = fnameArg.getValue();
  //  fname += (string)postfix;

  cout<<"Create C-ARM acquisition configuration file..."<<endl;
  cout<<"Configuration for sinogram simulation ? [Y/n]:";
  getline(cin, toto);
  if (toto == "n" || toto == "N")
    sgSim = false;
  
  cout<<"Equally distributed sources? [Y/n]:";
  getline(cin, toto);
  if (toto == "n" || toto == "N")
    eqSrc = false;
  
  cout<<"fanbeam [Y/n] :";
  getline(cin, toto);
  if (toto == "n" || toto == "N")
    fanbeam = false;

  cout<<"Total number of projections [180] :";
  do {
    getline(cin, toto);
    if (toto != "")
      nbProj = atoi(toto.data());
  } while (nbProj<=0);

  pSrc = new double[nbProj];

  if (eqSrc) {
    cout<<"Number of limited view intervals [1] :";
    do {
      getline(cin, toto);
      if (toto != "")
	limView = atoi(toto.data());
    }while (limView <=0);

    double *Views = new double[2*limView];
    double S = 0;
    for (size_t n=0; n<limView; n++) {
      // do {
      // 	printf("View interval No. %ld (a pair of number a<b) :", n);
      // 	cin>>Views[2*n]>>Views[2*n+1];
      // } while(Views[2*n]>=Views[2*n+1] || Views[2*n]<0 || Views[2*n+1]>2*M_PI);
      printf("View interval No. %ld (a pair of number a,b) :", n);
      cin>>Views[2*n]>>Views[2*n+1];
      S += fabs(Views[2*n+1] - Views[2*n]);
    }
    cin.ignore();

    double dv = S / nbProj;
    size_t nbPpv[limView];
    size_t prj = 0;
    for (size_t n=0; n<limView; n++) {
      nbPpv[n] = (size_t)ceil(fabs(Views[2*n+1] - Views[2*n]) / dv);
      for (size_t m=0; m<nbPpv[n]; m++) {
	pSrc[prj] = Views[2*n] + (Views[2*n+1] > Views[2*n] ? 1 : -1) * dv * m;
	//cout<<pSrc[prj]<<endl;
	if (prj<nbProj) prj++;
	else break;
      }
    }
    delete [] Views;
  }
  else {
    cout<<"Input source positions :";
    for (size_t n = 0; n<nbProj; n++)
      cin>>pSrc[n];
  }
  
  cout<<"Rotation center to source distance [80.] :";
  do {
    getline(cin, toto);
    if (toto != "")
      rSrc = atof(toto.data());
  } while (rSrc <=0);

  cout<<"Rotation center to detector distance [40.] :";
  do {
    getline(cin, toto);
    if (toto != "")
      rDet = atof(toto.data());
  } while (rDet <=0);

  cout<<"Detector size [50.] :";
  do {
    getline(cin, toto);
    if (toto != "")
      sizeDet = atof(toto.data());
  } while (sizeDet <=0);

  cout<<"Detector resolution (in pixel) [512] :";
  do {
    getline(cin, toto);
    if (toto != "")
      pixDet = atoi(toto.data());
  } while (pixDet <=0);
  // spDet = sizeDet / pixDet;
  // fout<<"spDet="<<spDet<<endl;

  // cout<<"Detector plan rotation angle wrt central axis [0.] :";
  // do {
  //   getline(cin, toto);
  //   if (toto != "")
  //     rtDet = atof(toto.data());
  // } while (rtDet <0);

  Array2d sizeObj = Acq::eval_sizeObj(fanbeam, rSrc, rDet, sizeDet);
  Array2u dimObj(512,512);
  if (!sgSim){
    printf("Object size in X axis [%f] :", sizeObj.x());
    do {
      getline(cin, toto);
      if (toto != "")
	sizeObj.x() = atof(toto.data());
    } while (sizeObj.x() <0);
    
    printf("Object size in Y axis [%f] :", sizeObj.y());
    do {
      getline(cin, toto);
      if (toto != "")
	sizeObj.y() = atof(toto.data());
    } while (sizeObj.y() <0);    

    int pix=512;
    printf("Screen display dimension of object in larger axis [%d] :", pix);
    do {
      getline(cin, toto);
      if (toto != "")
	pix = atoi(toto.data());
    } while (pix <0);    

    if (sizeObj.x()>=sizeObj.y()) {
      dimObj.x() = pix;
      dimObj.y() = (int)ceil(sizeObj.y() / sizeObj.x() * pix / 2) * 2;
    }
    else {
      dimObj.y() = pix;
      dimObj.x() = (int)ceil(sizeObj.x() / sizeObj.y() * pix / 2) * 2;
    }
  }

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

  if (!sgSim) {
    fout<<"sizeObj="<<sizeObj.x()<<", "<<sizeObj.y()<<endl;
    fout<<"dimObj="<<dimObj.x()<<", "<<dimObj.y()<<endl;
  }

  fout.close();
  cout<<"Acquisition configuration wrote into file : "<<fname<<endl;
  
  delete [] pSrc;
  return 0;
}

