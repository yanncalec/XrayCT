// Image reconstruction by L1 minimization method and blob representation

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

struct BlobL1_Parms {
  // Acquisition
  int nbProj;
  int pixDet;

  // BlobImage
  bool hex;
  double fspls;
  double cut_off_err;
  double fcut_off_err;
  int nbScale;
  double dil;
  string blobname;
  bool tightframe;

  // Projector
  bool stripint;
  bool tunning;

  // // Algo related
  bool besov;
  double nz;
  double nnz;
  bool debias;
  string algo;
  double mu;
  double mu_rel;

  double tol;
  int maxIter;
  int rwIter;
  double epsilon;
  double snr;
  double corr;
  double uqi;
  double mse;
  double si;  
  double time;
};

void save_algo_parms(const string &fname, const BlobL1_Parms &AP, const BlobImage &BI, const ConvMsg *msg);
void save_algo_parms_batch_csv(const string &fname, const BlobL1_Parms &AP, const BlobImage &BI);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by reweighted L1 minimization on multiscale blob image.", ' ', "0.1");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for reconstruction. [hexagonal grid]", cmd, false);
  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to ideal sampling step of reconstruction grid. [4]", false, 4, "double", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  TCLAP::SwitchArg notunningSwitch("", "notunning", "no normalization for projector [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-blob image model. [4]", false, 4, "int", cmd);
  //TCLAP::ValueArg<int> modelArg("", "model", "gauss-diffgauss multi-blob image model. [0]", false, 0, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, fixed to 2 for diffgauss blob. [2.0]", false, 2.0, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only for nbScale > 1 : 'diff', 'mexhat', 'd4gs'.[mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. Reduce this value (lower than 1e-4) is not recommended for diff-gauss blob. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-1]", false, 1e-1, "double", cmd);
  TCLAP::SwitchArg tightframeSwitch("", "tf", "use tight frame multiscale system [false]", cmd, false);
  TCLAP::SwitchArg besovSwitch("", "besov", "use Besov norm as initial reweighted l1-norm [false]", cmd, false);
  TCLAP::ValueArg<double> nzArg("", "nz", "The percentage (between 0 and 1) of the biggest coeffcients to be kept in the scale-product model. 0 or 1 turns off the constraint.  [0.05]", false, 0.05, "double", cmd);
  TCLAP::SwitchArg debiasSwitch("", "debias", "MSE debiasing on reconstruction using support information [false]", cmd, false);

  TCLAP::ValueArg<string> algoArg("", "algo","L1-minimization algorithm : ist, fista. [fista]",false,"fista","string", cmd);

  TCLAP::ValueArg<double> muArg("", "mu", "normalized penalty coefficient for data fidelity. [10]", false, 10, "double", cmd);
  TCLAP::ValueArg<int> maxIterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. [1e-4]", false, 1e-4, "double", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "reweighting parameter between (0, 1], meaningful for rwIter>1. Grid node of gradient norm bigger than epsilon is treated as contour, so small epsilon for strong edge detection/formation behavior. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string bcsv_fname = bcsv_fnameArg.getValue();
  string pfix = pfixArg.getValue();
  if (pfix == "")
    pfix = getenv ("SYSNAME");

  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();
  
  BlobL1_Parms AP;			// Algo structure
  AP.hex = !catSwitch.getValue();
  GridType gtype = (AP.hex)?_Hexagonal_ : _Cartesian_;
  AP.fspls = fsplsArg.getValue();

  AP.nbScale = nbScaleArg.getValue(); 
  AP.blobname = blobnameArg.getValue();
  AP.dil =  dilArg.getValue();

  if (AP.nbScale == 1) {
    AP.blobname = "gauss";
    AP.dil = 1;
  }
  else {
    if (AP.blobname == "diff")
      AP.dil = 2.0;
    assert(AP.dil > 1);
  }

  AP.cut_off_err = cut_off_errArg.getValue();
  AP.fcut_off_err = fcut_off_errArg.getValue();
  AP.tightframe = tightframeSwitch.getValue();
  AP.besov = besovSwitch.getValue();
  AP.nz = nzArg.getValue();
  AP.debias = debiasSwitch.getValue();
  int nterm = 0;

  AP.stripint = stripintSwitch.getValue();
  AP.tunning = !notunningSwitch.getValue();
  AP.algo = algoArg.getValue();

  AP.mu = muArg.getValue();
  AP.maxIter = maxIterArg.getValue();
  AP.tol = tolArg.getValue();

  AP.rwIter = rwIterArg.getValue();
  AP.epsilon = epsilonArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);
  
  // Create output path
  char buffer[256];
  if (pfix == "")
    sprintf(buffer, "BlobL1_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%1.1e]_nz[%1.1e]_tol[%1.1e]_rwIter[%d]_epsilon[%2.2f]_algo[%s]", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.mu, AP.nz, AP.tol, AP.rwIter, AP.epsilon, AP.algo.c_str());
  else
    sprintf(buffer, "BlobL1_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%1.1e]_nz[%1.1e]_tol[%1.1e]_rwIter[%d]_epsilon[%2.2f]_algo[%s].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.mu, AP.nz, AP.tol, AP.rwIter, AP.epsilon, AP.algo.c_str(), pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);
  AP.nbProj = conf.nbProj_total;
  AP.pixDet = conf.pixDet;

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);
  // normalize first Y by its maximum coefficient
  double Ynrml = Y.abs().maxCoeff(); // normalization factor to compensate the sinogram value impact
  Y /= Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its maximum coefficient : "<<Ynrml<<endl;

  // Pixel phantom image
  ImageXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  if (AP.blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "diff") {
    BI = BlobImageTools::MultiGaussDiffGauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  }
  else if (AP.blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err);
  else {
    cerr<<"Unknown blob profile!"<<endl;
    exit(0);
  }

  // Init GPU Blob projector
  BlobProjector *P;
  if (AP.stripint)
    P = new BlobProjector(conf, BI, 2);
  else {
    if (AP.blobname == "diff")	// always use table projector for diff-gauss blob
      P = new BlobProjector(conf, BI, 1); 
    else
      P = new BlobProjector(conf, BI, 0);
  }
  if (AP.tunning) {			// Normalize the projector
    if (verbose)
      cout<<"Estimation of operator norm..."<<endl;
    P->set_opNorm(P->estimate_opNorm());
  }

  if (verbose > 2) { 
    cout<<*BI<<endl;
    cout<<*P<<endl;
  }

  // Emperical value of optimal mu, it scales well for different data settings.
  //mu *= 3.25e7 / P->get_dimX() / (conf.pixDet*conf.nbProj_total); // emperical constant 32501022.72
  //mu *= 5e-5*(1.*conf.pixDet*conf.nbProj_total) / P->get_dimX();
  // mu *= 1e-7 * conf.pixDet*conf.nbProj_total*sqrt(log(P->get_dimX())); Suggested by Starck, dont work well
  AP.mu_rel = AP.mu;
  AP.mu /= P->get_dimX();

  // Initialization
  ArrayXd Xr, Xr0;// = P->backward(Y);
  Xr.setZero(P->get_dimX());

  ArrayXd W; 
  W.setOnes(Xr.size());

  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image

  ConvMsg* msg = new ConvMsg[AP.rwIter];
  clock_t t0 = clock();

  // The following is the reweighted iteration  
  {
    ArrayXd W = ArrayXd::Ones(Xr.size());			       // Reweighting coefficient
    if (AP.besov) {
      vector<ArrayXd> vW = BI->separate(W);
      double fc = 1;
      cout<<"MS blob scaling : "<<BI->get_scaling()<<endl;

      for (int s=vW.size()-1; s>=0; s--) {
	// for (int s=0; s<vW.size(); s++) {
	vW[s] *= fc;
	fc *= BI->get_scaling();
      }
      W = BI->joint(vW);		// Scale dependant reweighting
    }

    ArrayXd toto(Xr.size());
    ArrayXd Nu, Lambda;		// Meaningful only for L1AL3 method
    ArrayXd S(Xr.size());	// Support of best approximation

    double nz0 = AP.nz / AP.rwIter;
    for(int n=0; n<AP.rwIter; n++) {
      if (verbose)
	cout<<"\nReweighted L1 minimization iteration : "<<n<<endl;    

      //Xr.setZero();		// Clean first Xr

      if (AP.algo == "fista")
	msg[n] = SpAlgo::L1FISTA(*P, W, Y, Xr, S, (n+1)*AP.nz/AP.rwIter, AP.mu, AP.tol, AP.maxIter, verbose);
      else if (AP.algo == "ist")
	msg[n] = SpAlgo::L1IST(*P, W, Y, Xr, S, (n+1)*AP.nz/AP.rwIter, AP.mu, AP.tol, AP.maxIter, verbose);

      // Debiasing
      Xr0 = Xr;
      if (AP.debias) {
	if (verbose) 
	  cout<<"Debiasing by Conjugate Gradient method..."<<endl;
	LinSolver::Solver_CG(*P, Y, Xr, 100, 1e-4, &S, false);
      }

      msg[n].norm = Xr.abs().sum();
      msg[n].res = (P->forward(Xr)-Y).matrix().norm();
      msg[n].vobj = 0.5 * msg[n].res * msg[n].res + AP.mu * msg[n].norm;    
      
      W = 1 - S;
      // ArrayXd Mask = 1 - BI->prod_mask(Xr, nz); // Attention : The mask is inverse, 0 for the presence of blob
      // // vector<ArrayXd> V = BI->separate(Mask);
      // // for (int m=0; m<V.size(); m++) {
      // //  	//printf("Scale %d, min = %e, max = %e\n", m, V[m].minCoeff(), V[m].maxCoeff());
      // // 	CmdTools::imshow(V[m], BI->bgrid[m]->vshape.y(), BI->bgrid[m]->vshape.x(), "Mask");	
      // // }

      // double xmax = Xr.abs().maxCoeff();
      // for (size_t m=0; m<Xr.size(); m++) {
      // 	toto[m] = 1/(fabs(Xr[m]) + epsilon * xmax);
      // }
      //  W = toto * Mask / toto.maxCoeff();
      // cout<<"Wmax = "<<W.maxCoeff()<<endl;
      // cout<<"Wmin = "<<W.minCoeff()<<endl;

      //Xr *= (1-Mask);

      if (verbose > 1) {	      
	sprintf(buffer, "%s/rwIter_%d", outpath_fname.c_str(), n);
	ImageXXd imr = BI->blob2pixel(Xr, conf.dimObj);
	CmdTools::imsave(imr, buffer);
	//CmdTools::savearray(imr, buffer);
	// sprintf(buffer, "%s/rwIter_debiased_%d", outpath_fname.c_str(), n);
	// imr = BI->blob2pixel(Xd, conf.dimObj);
	// CmdTools::imsave(imr, buffer, false);
      }
    }
  }

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization and descaling Xr
  Xr *= (AP.tunning) ? (Ynrml / P->get_opNorm()) : Ynrml;

  Imr = BI->blob2multipixel(Xr, conf.dimObj);
  imr = Tools::multi_imsum(Imr);
  if (conf.phantom != "") {
    AP.nnz = Tools::l0norm(Xr) *1./ Xr.size();
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("NNZ = %f\tUQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.nnz, AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);
  if (AP.debias) {
    sprintf(buffer, "%s/xr0", outpath_fname.c_str()); // non debiased data
    CmdTools::savearray(Xr0, buffer);
  }
  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi");
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, *BI, msg);
  if (bcsv_fname != "") {
    save_algo_parms_batch_csv(bcsv_fname, AP, *BI);
  }
  // Save reconstructed image(s)
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);
  if (AP.nbScale > 1)
    CmdTools::multi_imsave(Imr, buffer);

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;

  if (verbose > 1) {
    CmdTools::imshow(imr, "Reconstructed blob image");
    if (AP.nbScale > 1)
      CmdTools::multi_imshow(Imr, "Reconstructed blob image");
  }

  return 0;
}


void save_algo_parms(const string &fname, const BlobL1_Parms &AP, const BlobImage &BI, const ConvMsg *msg)
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
  fout<<BI<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"stripint="<<AP.stripint<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"algo="<<AP.algo<<endl;
  fout<<"nz="<<AP.nz<<endl;
  fout<<"besov="<<AP.besov<<endl;
  fout<<"debias="<<AP.debias<<endl;
  fout<<"mu="<<AP.mu<<endl;
  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;

  fout<<"rwIter="<<AP.rwIter<<endl;
  fout<<"epsilon="<<AP.epsilon<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"nnz="<<AP.nnz<<endl;
  fout<<"snr="<<AP.snr<<endl;
  fout<<"corr="<<AP.corr<<endl;
  fout<<"uqi="<<AP.uqi<<endl;
  fout<<"mse="<<AP.mse<<endl;
  fout<<"si="<<AP.si<<endl;
  fout<<"time="<<AP.time<<endl;

  for (int n=0; n<AP.rwIter; n++) {
    fout<<endl<<"Reweighted iteration : "<<n<<endl;
    fout<<"niter="<<msg[n].niter<<endl;
    fout<<"res="<<msg[n].res<<endl;
    fout<<"l1norm="<<msg[n].norm<<endl;
  }

  fout.close();
}

void save_algo_parms_batch_csv(const string &fname, const BlobL1_Parms &AP, const BlobImage &BI)
{
  ofstream fp;
  fp.open(fname.data(), ios::app | ios::out);
  if (!fp) {
    cout <<"Cannot open file : "<<fname<<endl;
    exit(1);
  }
  long begin, end;
  fp.seekp(0, ios_base::beg); // put position pointer to the end
  begin = fp.tellp();
  fp.seekp(0, ios_base::end); // put position pointer to the end
  end = fp.tellp();
  char buffer[256];

  if (begin==end) {
    //cout<<"file is empty"<<endl;
    fp<<"BlobL1_batch_reconstruction_results"<<endl;

    // BlobImage
    fp<<"nbProj, "<<"pixDet, "<<"fspls, "<<"nbNode, ";
    for (int n=0; n<BI.get_nbScale(); n++) {
      sprintf(buffer, "blob_radius_%d, bgrid_splStep_%d, ",n, n);
      fp<<buffer;
    }

    // Algo related
    fp<<"mu_rel, nnz, snr, corr, uqi, mse, si, time"<<endl;
  }

  //fp<<AP.hex<<", "<<AP.fspls<<", "<<AP.upSpl<<", "<<AP.cut_off_err<<", "<<AP.fcut_off_err<<", "<<AP.nbScale<<", "<<AP.dil<<", "<<AP.blob_name<<", ";
  fp<<AP.nbProj<<", "<<AP.pixDet<<", "<<AP.fspls<<", "<<BI.get_nbNode()<<", ";
  for (int n=0; n<BI.get_nbScale(); n++) {
    fp<<BI.blob[n]->radius<<", "<<BI.bgrid[n]->splStep<<", ";
  }
  //fp<<AP.stripint<<", "<<AP.tunning<<", ";
  fp<<AP.mu_rel<<", "<<AP.nnz<<", "<<AP.snr<<", "<<AP.corr<<", "<<AP.uqi<<", "<<AP.mse<<", "<<AP.si<<", "<<AP.time;
  fp<<endl;

  fp.close();  
}
