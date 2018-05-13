// Debias multiscale blob reconstructions

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, bool stripint, 
		     double spa,double tol, int maxIter, 
		     double snr, double corr, double res,
		     double time, const ConvMsg &msg);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("Debiase a multiscale BlobImage object.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name [auto.]",false,"","string", cmd);

  TCLAP::ValueArg<size_t> dimArg("", "dim", "Resolution of the larger side of interpolation region . [512]", false, 512, "size_t", cmd);
  TCLAP::ValueArg<string> xr_fnameArg("", "xr","Input blob coefficient file name",false,"xr.dat","string", cmd);
  TCLAP::ValueArg<string> bi_fnameArg("", "bi","Input BlobImage object file name",false,"bi.dat","string", cmd);

  //TCLAP::SwitchArg debiasingSwitch("", "debiasing", "Debiasing of blob coefficients by interscale product and CG. [false]", cmd, false);
  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"","string", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  TCLAP::ValueArg<double> spaArg("", "spa", "The percentage (between 0 and 1) of the biggest coeffcients to be kept in the scale-product model. 0 or 1 turns off the constraint.  [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> sclArg("", "scl", "scaling factor btw scales.  [0, use default value of blob image]", false, 0, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "CG iterations stopping tolerance. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of CG iterations. [100]", false, 100, "size_t", cmd);

  TCLAP::ValueArg<size_t> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "size_t", cmd);
  TCLAP::SwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd, false);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();

  string xr_fname = xr_fnameArg.getValue();
  string bi_fname = bi_fnameArg.getValue();

  string cfg_fname = cfg_fnameArg.getValue();
  string sg_fname = sg_fnameArg.getValue();
  double spa = spaArg.getValue();
  double scl = sclArg.getValue();
  bool stripint = stripintSwitch.getValue();
  double tol = tolArg.getValue();
  int maxIter = IterArg.getValue();

  bool verbose = verboseSwitch.getValue();
  size_t gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  // Create output path
  char buffer[256];
  sprintf(buffer, "Debiasing_spa[%1.3f]_tol[%1.1e]_Iter[%d]", spa, tol, maxIter);
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // Load BlobImage
  BlobImage *BI = BlobImageTools::load(inpath_fname+"/"+bi_fname);
  // Load blob coefficients
  ArrayXd Xr = CmdTools::loadarray(inpath_fname+"/"+xr_fname, BI->get_nbNode());
  
  // ArrayXi Idx = (Xr.abs()>0).select(1, ArrayXi::Zero(Xr.size()));
  // cout<<"Non zero terms : "<<Idx.sum()<<endl;
  // cout<<"Percentage : "<<Idx.sum()*1./Idx.size()<<endl;

  spa = (fabs(spa)>1)? 1 : fabs(spa);
  if (spa > 0) {
    cout<<"Sparsity before thresholding : "<<endl;
    BI->sparsity(Xr);

    ArrayXi PSupp = BI->scalewise_prodmask(Xr, scl, spa);
    Xr *= (PSupp>0).select(1. , ArrayXd::Zero(PSupp.size()));

    cout<<"Sparsity after thresholding : "<<endl;
    BI->sparsity(Xr);
  }

  if (cfg_fname == "") {
    cfg_fname = inpath_fname + "/../" + "acq.cfg";
  }
  if (sg_fname == "") {
    sg_fname = inpath_fname + "/../" + "sino.dat";
  }

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);

  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);
  // Y /= Y.abs().maxCoeff(); // Useless?

  // Pixel phantom image
  ImageXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobProjector *P;
  if (stripint)
    P = new BlobProjector(conf, BI, 2);
  else {
    if (BI->blob[1]->blobname == "diff")	// always use table projector for diff-gauss blob
      P = new BlobProjector(conf, BI, 1); 
    else
      P = new BlobProjector(conf, BI, 0);
  }

  clock_t t0 = clock(); // Reconstruction time
  ConvMsg msg = LinSolver::Projector_Debiasing(*P, Y, Xr, maxIter, tol, verbose);

  double rectime = (clock() - t0) / (double)CLOCKS_PER_SEC;
  printf("Debiasing taken %lf seconds\n", rectime);     

  double res = (P->forward(Xr)-Y).matrix().norm();
  printf("Residual |Ax-y|=%f\n", res);
  // Denormalization and descaling Xr
  //Xr *= (tunning) ? (1. / P->get_opNorm()) : 1;

  double recsnr=0, reccorr=0;	// SNR and Corr of reconstruction
  vector<ImageXXd> Imr = BI->blob2multipixel(Xr, conf.dimObj); // all scales image
  ImageXXd imr = Tools::multi_imsum(Imr);
  if (conf.phantom != "") {
    double recsnr = Tools::SNR(imr, im);
    double reccorr = Tools::Corr(imr, im);
    double uqi = Tools::UQI(imr, im);
    double mse = (imr-im).matrix().squaredNorm() / im.size();
    double si = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", uqi, mse, reccorr, recsnr, si);
  }

  // Save reconstructed coefficients Xr
  CmdTools::savearray(Xr, outpath_fname+"/xr");
  // Save reconstructed image(s)
  CmdTools::imsave(imr, outpath_fname+"/debiased");
  if (BI->get_nbScale() > 1)
    CmdTools::multi_imsave(Imr, outpath_fname+"/debiased");
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", stripint, 
		  spa, tol, maxIter,
		  recsnr, reccorr, res,
		  rectime, msg);

  if (verbose) {
    CmdTools::imshow(imr, "Debiased");
    CmdTools::multi_imshow(Imr, "Debiased");
  }
    
  cout<<"Outputs saved in directory "<<inpath_fname<<endl;
  return 0;
}

void save_algo_parms(const string &fname, bool stripint, 
		     double spa, double tol, int maxIter, 
		     double snr, double corr, double res, 
		     double time, const ConvMsg &msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[Projector Parameters]"<<endl;
  fout<<"raytracing="<<!stripint<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"maxIter="<<maxIter<<endl;
  fout<<"tol="<<tol<<endl;

  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"snr="<<snr<<endl;
  fout<<"corr="<<corr<<endl;
  fout<<"time="<<time<<endl;

  fout<<"niter="<<msg.niter<<endl;
  fout<<"CG res="<<msg.res<<endl;
  fout<<"residual |Ax-y|="<<res<<endl;

  fout.close();
}

