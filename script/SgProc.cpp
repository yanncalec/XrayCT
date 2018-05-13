// Sinogram processing tools

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

int main(int argc, char **argv) {

  // Load configuration
  TCLAP::CmdLine cmd("Basic sinogram processing tool. This program can add noise to a noiseless sinogram, perform wavelet denoising, average detector pixels or extract a projection subset of sinogram.\n\
Examples:\n\
1. SgProc.run DIR_IN [DIR_OUT] --nbProj 64 --pixDet 256\n\
extracts a sub-sinogram of 64 projections and 256 detector pixels.\n\
2. SgProc.run DIR_IN [DIR_OUT] --float --nbProj 64 --pixDet 256 --log\n\
loads the binary single precision data, float extracts a sub-sinogram of 64 projections and 256 detector pixels, furthermore take logarithm transform on the sinogram.\n\
3. SgProc.run DIR_IN [DIR_OUT] --nbProj 64 --pixDet 256 --log --rvDet\n\
reverses the order of detector pixel indexes.");

  //TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name, it contains the processed sinogram and updated configuration file. By default a sub-directory is created under input path.",false,"","string", cmd);
  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);
  TCLAP::SwitchArg imageSwitch("", "image", "Sinogram is a image file. [false]", cmd, false);
  TCLAP::SwitchArg logSwitch("", "log", "Sinogram is photon data, taking log transform. [false]", cmd, false);
  TCLAP::SwitchArg negSwitch("", "neg", "Sinogram is inverse data, taking 1-minus transform. [false]", cmd, false);
  TCLAP::ValueArg<double> zpArg("", "zp", "Zero-padding factor of sinogram. [1]", false, 1, "double", cmd);
  TCLAP::SwitchArg floatSwitch("", "float", "Array is single precision float. [false]", cmd, false);
  TCLAP::SwitchArg endianSwitch("", "endian", "Swap the endian format of array. [false]", cmd, false);

  //  TCLAP::ValueArg<string> opArg("", "op","Operation type : proj, rebin, noise, denoise.\nproj : Extract a subset of projection.\nrebin : Rebin the detector to a desired resolution by averaging detector pixels.\nnoise : Add noise to sinogram.\ndenoise : perform wavelet denoising.",true,"proj","string", cmd);
  TCLAP::ValueArg<size_t> nbProjArg("", "nbProj", "Number of equi-distributed subset of projections to be extracted. [0]", false, 0, "size_t", cmd);
  TCLAP::SwitchArg endpointSwitch("", "ep", "Keep the original first and the last when extracting projections. [false]", cmd, false);
  TCLAP::ValueArg<size_t> pixDetArg("", "pixDet", "Detector resolution to be extracted. [0]", false, 0, "size_t", cmd);
  TCLAP::ValueArg<int> detOffsetArg("", "doff", "Detector offset in number of pixelsm. The 2D sinogram index is translated in detector direction by this value. [0]", false, 0, "size_t", cmd);
  TCLAP::SwitchArg rvDetSwitch("", "rvDet", "Reverse detector pixel order. [false]", cmd, false);
  TCLAP::ValueArg<size_t> phsArg("", "phs", "Source intensity, 0 for noiseless sinogram. 1e7 for ~ 60db. [0]", false, 0, "size_t", cmd);
  TCLAP::ValueArg<double> snrArg("", "snr", "Desired SNR of sinogram. Gaussian noise model will be used if this value > 0. [0]", false, 0, "double", cmd);
  TCLAP::ValueArg<double> denoiseArg("", "denoise", "Noise level in wavelet denoising. [0]", false, 0, "double", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [false]", cmd, false);

  cmd.parse(argc, argv);

  //string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  if (outpath_fname=="") outpath_fname = "./";

  string cfg_fname = cfg_fnameArg.getValue();
  string sg_fname = sg_fnameArg.getValue();

  bool image = imageSwitch.getValue();
  bool logflag = logSwitch.getValue();
  bool negflag = negSwitch.getValue();
  double zp = zpArg.getValue();
  bool floatflag = floatSwitch.getValue();
  bool endianflag = endianSwitch.getValue();

  size_t nbProj = nbProjArg.getValue();
  bool endpoint = endpointSwitch.getValue();
  size_t pixDet = pixDetArg.getValue();
  size_t phs = phsArg.getValue();  
  double snr = snrArg.getValue();
  int doff = detOffsetArg.getValue();
  bool rvDet = rvDetSwitch.getValue();
  double denoise = denoiseArg.getValue();
  int verbose = verboseSwitch.getValue();

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);
  // Load sinogram
  ArrayXd Y;
  if (image) {
    ImageXXd toto = CmdTools::imread(sg_fname);
    Y = Map<ArrayXd>(toto.data(), toto.size());
  }
  else
    Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet, floatflag, endianflag);

  if (verbose)
    CmdTools::imshow(Y, conf.nbProj_total, conf.pixDet, "Origin sinogram");

  // Logarithm or negativity transform of sinogram
  if (logflag) {
    Y /= Y.maxCoeff();
    ArrayXb Idx = (Y <= 0).select(true, ArrayXb::Constant(Y.size(), false)); // the negative and zero value indexes
    ArrayXd toto = (Y <= 0).select(-1, Y.log().abs()); // take log on positive value indexes
    //ArrayXd toto1 = (Idx).select(0, toto);
    double vmax = toto.maxCoeff();
    for (int n= 0; n<Y.size(); n++)
      if (Y[n]>0) vmax = fmax(vmax, toto[n]);

    Y = (Idx).select(vmax, toto);
  }
  else if (negflag) {
    Y = Y.maxCoeff() - Y;
  }

  // Reverse the detector pixel order
  if(rvDet) {
    ArrayXd Y0=Y;
    Y.setZero();
    for(int p=0; p<conf.nbProj_total; p++) {
      for (int n=0; n<conf.pixDet; n++)
	Y[p*conf.pixDet + n] = Y0[p*conf.pixDet + conf.pixDet-1-n];
    }
  }

  // Detector pixel offset
  if(doff!=0){
    //cout<<doff<<endl;
    assert(abs(doff)<conf.pixDet);
    ArrayXd Y0=Y;
    Y.setZero();
    for(int p=0; p<conf.nbProj_total; p++) {
      if(doff>0)
	Y.segment(p*conf.pixDet+doff, conf.pixDet-doff) = Y0.segment(p*conf.pixDet, conf.pixDet-doff);
      else
	Y.segment(p*conf.pixDet, conf.pixDet+doff) = Y0.segment(p*conf.pixDet-doff, conf.pixDet+doff);
    }
  }

  // Zero-padding of sinogram
  int pixDet0 = conf.pixDet;
  if(zp>1){
    // Adjust the acq system size
    conf.pixDet = (int)ceil(zp * conf.pixDet);
    conf.sizeDet = (conf.sizeDet / pixDet0) * conf.pixDet;
    //conf.diamFOV = Acq::eval_diamFOV(conf.fanbeam, conf.rSrc, conf.rDet, conf.sizeDet);
    conf.diamFOV *= zp;
    assert(conf.diamFOV/2 <conf.rSrc.minCoeff());
  }

  if(zp>1) {
    int df = conf.pixDet - pixDet0;
    int nhead = (int)ceil(df/2);
    //int nbutt = df - nhead;
    ArrayXd Y0=Y;
    Y.setZero(conf.pixDet * conf.nbProj_total);
    for(int p=0; p<conf.nbProj_total; p++) {
      Y.segment(p*conf.pixDet+nhead, pixDet0) = Y0.segment(p*pixDet0, pixDet0);
    }
  }

  ArrayXd Z;
  nbProj = (nbProj == 0) ? conf.nbProj_total : min(conf.nbProj_total, nbProj);
    
  if (nbProj == conf.nbProj_total) {
    Z = Y;
  }
  else {
    Z = ArrayXd::Zero(nbProj * conf.pixDet);
    // int sp = (int)floor(conf.nbProj_total * 1. / (nbProj+1)); // spacing 
    // ArrayXd toto = ArrayXd::LinSpaced(0, conf.nbProj_total-sp, nbProj);
    // ArrayXi idx = toto.cast<int>();
    ArrayXi idx = Tools::subseq(conf.nbProj_total, nbProj, endpoint);
    //cout<<idx.transpose()<<endl;

    for (int p=0; p<nbProj; p++) {
      Z.segment(p*conf.pixDet, conf.pixDet) = Y.segment(idx[p]*conf.pixDet, conf.pixDet);
    }
    conf = CmdTools::extract_config(conf, nbProj, endpoint);
    //cout<<nbProj<<" "<<conf.nbProj_total<<endl;
  }

  pixDet = (pixDet == 0) ? conf.pixDet : min(conf.pixDet, pixDet);
  if (pixDet < conf.pixDet) {
    int RB = (int)ceil(conf.pixDet * 1. / pixDet); // Rebinning factor
    ArrayXd U(pixDet*conf.nbProj_total); U.setZero();
    ArrayXd Z0; 
    ArrayXi idx = Tools::subseq(conf.pixDet, pixDet);

    for (int p=0; p<conf.nbProj_total; p++) { // conf.nbProj_total has already been modified in "conf = CmdTools::extract_config(conf, nbProj, endpoint);"
      Z0.setZero(conf.pixDet + 3*RB);
      Z0.head(conf.pixDet) = Z.segment(p*conf.pixDet, conf.pixDet);      
      //cout<<Z0.size()<<endl;
      for (int n=0; n<pixDet; n++)
	U[p*pixDet+n] = Z0.segment(idx[n], RB).mean(); // Taking the mean to reduce noise and equalize the detector
    }

    Z = U;
    conf.pixDet = pixDet;
    conf.spDet = conf.sizeDet/conf.pixDet;
  }

  char buffer[256];
  sprintf(buffer,"%s/nbProj%d_pixDet%d", outpath_fname.c_str(), conf.nbProj_total, conf.pixDet);
  if (denoise > 0) {
    cerr<<"Denoising not implemented yet!"<<endl;
    exit(1);
    //sprintf(buffer,"%s/nbProj[%d]_pixDet[%d]_denoise", inpath_fname.c_str(), conf.nbProj_total, conf.pixDet);
  }
  else if (snr>0 || phs>0) {
    Array3d info(0,0,0); // Noisy sinogram information
    if (snr > 0) {
      Z = Tools::gaussian_data(Z, snr, info);
      cout<<"Gaussian noisy sinogram : "<<endl;
      cout<<"Noise energy : "<<info[0]<<", per detector : "<<info[1]<<endl;
      cout<<"SNR : "<<info[2]<<" db"<<endl;
    }
    else if (phs > 0) {
      Z = Tools::poisson_data(Z, phs, info);
      cout<<"Poisson noisy sinogram : "<<endl;
      cout<<"Noise energy : "<<info[0]<<", per detector : "<<info[1]<<endl;
      cout<<"SNR : "<<info[2]<<" db"<<endl;
      conf.phs = phs;
    }

    conf.noise_energy = info[0];
    conf.noise_std = info[1];
    conf.snr = info[2];    
    sprintf(buffer,"%s/proj%d_pixDet%d_noisy", outpath_fname.c_str(), conf.nbProj_total, conf.pixDet);
  }

  if (verbose) {
    CmdTools::imshow(Z, conf.nbProj_total, conf.pixDet, "Processed sinogram");
  }

  outpath_fname = CmdTools::creat_outpath(buffer);

  // save sinogram data, configuration into output directory
  CmdTools::save_acqcarm(conf, outpath_fname+"/acq.cfg");
  CmdTools::savearray(Z, outpath_fname+"/sino");
  CmdTools::imsave(Z,  conf.nbProj_total, conf.pixDet, outpath_fname+"/sino");
  cout<<"Outputs saved in directory "<<outpath_fname<<endl;

  return 0;
}
