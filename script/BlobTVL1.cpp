// Image reconstruction by TV-L1 minimization method and blob
// representation using prxm (proximal) algorithm by default (spsl not
// tested). Experiences show that the convergence happens only after
// more than 2000 iterations.

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg msg);
void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP, const BlobImage &BI);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by TV-L1 minimization on multiscale blob image.\
Solve one of the following optimization problems:\n\
     min_f mu*|Af-b|^2 + (1-gamma)*TV(f)+gamma*|f|_1, (TVL1)\n\
     min_f (1-gamma)*TV(f)+gamma*|f|_1, st. Af=b (TVL1-EQ)\n\
     min_f (1-gamma)*TV(f)+gamma*|f|_1, st. |Af-b|^2<M*epsilon^2, (TVL1-DN).");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for blob image and reconstruction. The hexagonal grid requires less samples (86.6%) than cartesian grid for representing a bandlimited function. [hexagonal grid]", cmd, false);

  TCLAP::ValueArg<double> roiArg("", "roi", "factor to effective reconstruction ROI. [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. By default(roi=1, box=1), the square ROI is the smallest square including circle ROI. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to reconstruction bandwidth. [2]", false, 2, "double", cmd);
  TCLAP::ValueArg<int> upSplArg("", "upSpl", "Up-sampling factor for interpolation grid. [1]", false, 1, "int", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. The strip-integral is a more [false]", cmd, false);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);
  //TCLAP::SwitchArg notunningSwitch("", "notunning", "no normalization for projector [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-scale blob image model. nbScale=1 for gaussian blob image. [4]", false, 4, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, meaningful only when nbScale>1. [2]", false, 2, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only when nbScale>1. Possible choices : 'mexhat', 'd4gs'. [mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-1]", false, 1e-1, "double", cmd);
  TCLAP::SwitchArg frameSwitch("", "frame", "use frame but not tight frame multiscale system [false]", cmd, false);

  TCLAP::ValueArg<string> algoArg("", "algo", "algorithm for TVL1: prxm, spl. [prxm]", false, "prxm", "string", cmd);
  TCLAP::SwitchArg anisoSwitch("", "aniso", "anisotropic TV model. [false]", cmd, false);
  TCLAP::ValueArg<double> gammaArg("", "gamma", "L1 weight. [0.99]", false, 0.99, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "factor to the data fidelity. [2.5e6]", false, 2.5e6, "double", cmd);
  TCLAP::ValueArg<double> pgrArg("", "pgr", "Prxm: gradient penalty. [100]", false, 100, "double", cmd);
  TCLAP::ValueArg<double> pidArg("", "pid", "Prxm: identity penalty. This is in fact a factor, enlarge it when non convergence occurs. Small value accelerates convergence. [5]", false, 5, "double", cmd);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "Prxm: model selection: <0 for unconstraint TVL1, 0 for TVL1-EQ, >0 for TVL1-DN with epsilon as noise level per detector", false, -1, "double", cmd);
  TCLAP::ValueArg<double> tau1Arg("", "tau1", "SPL: tau1, must be sufficiently small.", false, 1e-6, "double", cmd);
  TCLAP::ValueArg<double> tau2Arg("", "tau2", "SPL: tau2, must be sufficiently small.", false, 1e-6, "double", cmd);

  TCLAP::ValueArg<double> tolArg("", "tol", "iteration stopping tolerance. Decrease it for high precision convergence. [1e-6]", false, 1e-6, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [5000]", false, 5000, "int", cmd);
  // TCLAP::ValueArg<double> cgtolArg("", "cgtol", "CG iterations stopping tolerance. [1e-3]", false, 1e-3, "double", cmd);
  // TCLAP::ValueArg<int> cgIterArg("", "cgiter", "maximum number of CG iterations. [50]", false, 50, "int", cmd);
  TCLAP::ValueArg<double> stolArg("", "stol", "support changes stopping tolerance. [5e-3]", false, 5e-3, "double", cmd);
  TCLAP::ValueArg<int> debiasArg("", "debias", "number cg iteration for debiasing. [25]", false, 25, "int", cmd);
  TCLAP::ValueArg<int> hIterArg("", "hIter", "number of homotopy continuation on mu. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> incrArg("", "incr", "homotopy continuation incremental factor. [0.5]", false, 0.5, "double", cmd);
  TCLAP::ValueArg<double> spaArg("", "spa", "The percentage (between 0 and 1) of the biggest coeffcients. This defines a support set S and the algorithm exits when S is stablized. sp=0 or 1 turns off the constraint.  [0]", false, 0, "double", cmd);
  TCLAP::SwitchArg noapprxSwitch("", "noapprx", "if true the L1 problem is solved perfectly at last homotopy iteration. [false]", cmd, false);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);
  TCLAP::ValueArg<int> FreqArg("", "mfreq", "Message print frequence. [50]", false, 50, "int", cmd);
  TCLAP::SwitchArg tmpSwitch("", "tmp", "if true the intermediary convergency results are saved. [false]", cmd, false);

  cmd.parse(argc, argv);

  AlgoParms AP;			// Algo structure
  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string bcsv_fname = bcsv_fnameArg.getValue();
  string pfix = getenv ("SYSNAME");

  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();

  AP.hex = !catSwitch.getValue();
  AP.gtype = (AP.hex)?_Hexagonal_ : _Cartesian_;

  AP.roi = roiArg.getValue();
  AP.box = boxArg.getValue();

  AP.sgNrml = !nosgNrmlSwitch.getValue();
  AP.fspls = fsplsArg.getValue();
  AP.upSpl = upSplArg.getValue();
  AP.stripint = stripintSwitch.getValue();

  AP.nbScale = nbScaleArg.getValue(); 
  AP.blobname = blobnameArg.getValue();
  AP.dil =  dilArg.getValue();
  
  if (AP.nbScale == 1) {
    AP.blobname = "gauss";
    AP.dil = 1;
  }
  else {
    assert(AP.blobname == "mexhat" || AP.blobname == "d4gs");
    assert(AP.dil > 1);
  }

  AP.cut_off_err = cut_off_errArg.getValue();
  AP.fcut_off_err = fcut_off_errArg.getValue();
  AP.tightframe = !frameSwitch.getValue();

  AP.algo = algoArg.getValue();
  AP.aniso = anisoSwitch.getValue();
  AP.mu = muArg.getValue();
  AP.gamma = gammaArg.getValue();
  AP.penalty_grad = pgrArg.getValue();
  AP.penalty_id = pidArg.getValue();
  AP.epsilon = epsilonArg.getValue();
  AP.tau1 = tau1Arg.getValue();
  AP.tau2 = tau2Arg.getValue();

  AP.maxIter = IterArg.getValue();
  AP.tol = tolArg.getValue();
  AP.hIter = hIterArg.getValue();
  AP.incr = incrArg.getValue();
  AP.stol = stolArg.getValue();
  AP.cgmaxIter = debiasArg.getValue();
  AP.spa = spaArg.getValue();
  AP.noapprx = noapprxSwitch.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();
  int Freq = FreqArg.getValue();
  bool tmp = tmpSwitch.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, AP.box, AP.roi);
  AP.nbProj = conf.nbProj_total;
  AP.pixDet = conf.pixDet;

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);    
  AP.Ynrml = (AP.sgNrml) ? Y.abs().mean() : 1.; // normalization factor to compensate the sinogram value impact
  Y /= AP.Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its mean abs value: "<<AP.Ynrml<<endl;

  // Pixel phantom image
  ImageXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  if (AP.blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err);
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

  Grid *igrid = new Grid(BI->bgrid[0]->sizeObj, BI->bgrid[AP.nbScale-1]->vshape*AP.upSpl, BI->bgrid[AP.nbScale-1]->splStep/AP.upSpl, AP.gtype, BI->bgrid[0]->diamROI); // Interpolation grid

  // Init GPU Blob projector
  BlobProjector *P;
  if (AP.stripint)
    P = new BlobProjector(conf, BI, 2);
  else
    P = new BlobProjector(conf, BI, 0);

  // Gradient operator
  BlobInterpl *G = new BlobInterpl(BI, igrid, _Grad_);
  if (verbose > 2) {
    cout<<*P<<endl;
    cout<<*G<<endl;
  }

  // Normalization of mu
  if (AP.epsilon<0) {
    //double mu_factor = igrid->nbNode / pow(1.*STD_DIAMFOV, 2.) / (1.* conf.nbProj_total * conf.pixDet);
    double mu_factor = (conf.pixDet * conf.nbProj_total) * log(P->get_dimX());
    cout<<"Scaling mu by the factor: "<<mu_factor<<endl;
    AP.mu_rel = AP.mu;
    AP.mu /= mu_factor;
  }
  else {
    cout<<"Constraint TV-L1 minimization model: mu should be set to the same order as gradient penalty!"<<endl;
    AP.mu = AP.penalty_grad;
  }
  // Set automatically the identity penality AP.penalty_id
  // In principle we should use:
  // AP.penalty_id = cst * AP.mu; then 1/cst is the gradient step allowing convergence
  // But in reality this won't work unless cst >= 1e7
  AP.penalty_id *= AP.mu_rel; // This works well

  // Initialization
  //ArrayXd Xr = P->backward(Y);
  ArrayXd Xr = ArrayXd::Zero(P->get_dimX());
  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image

  // Create output path
  char buffer[256];
  if (AP.hIter>1) {
    if (AP.epsilon<0)
      sprintf(buffer, "BlobTVL1_fspls[%1.1e]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%1.1e]_gamma[%1.1e]_tol[%1.1e]_hIter[%d]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.mu_rel, AP.gamma, AP.tol, AP.hIter, AP.roi, AP.box, pfix.c_str());
    else
      sprintf(buffer, "BlobTVL1_fspls[%1.1e]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_gamma[%1.1e]_epsilon[%1.1e]_tol[%1.1e]_hIter[%d]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.gamma, AP.epsilon, AP.tol, AP.hIter, AP.roi, AP.box, pfix.c_str());
  }
  else {
    if (AP.epsilon<0)
      sprintf(buffer, "BlobTVL1_fspls[%1.1e]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%1.1e]_gamma[%1.1e]_tol[%1.1e]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.mu_rel, AP.gamma, AP.tol, AP.roi, AP.box, pfix.c_str());
    else
      sprintf(buffer, "BlobTVL1_fspls[%1.1e]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_gamma[%1.1e]_epsilon[%1.1e]_tol[%1.1e]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.gamma, AP.epsilon, AP.tol, AP.roi, AP.box, pfix.c_str());
  }

  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  ConvMsg msg;
  clock_t t0 = clock();  

  if (AP.algo == "spl")
    msg = SpAlgo::TVL1SPL(*P, *G, Y, Xr, 
  			  AP.aniso, (1-AP.gamma)/AP.mu, AP.gamma/AP.mu, 
  			  AP.tau1, AP.tau2, AP.tol, AP.maxIter, 
  			  AP.stol, AP.spa, verbose);
  else if (AP.algo == "prxm") {
    if (AP.hIter>1)
      msg = SpAlgo::TVL1Prxm_Homotopy(*P, *G, Y, Xr,
				      AP.gamma, AP.mu, AP.penalty_grad, AP.penalty_id, AP.aniso,
				      AP.tol, AP.maxIter, AP.stol, AP.spa, 
				      AP.hIter, AP.incr, AP.cgmaxIter, verbose, AP.noapprx, Freq);
    else
      msg = SpAlgo::TVL1Prxm(*P, *G, Y, Xr, 			   
			     AP.epsilon*sqrt(conf.nbProj_total*conf.pixDet), AP.gamma, 
			     AP.mu, AP.penalty_grad, AP.penalty_id, AP.aniso,
			     AP.tol, AP.maxIter,
			     0, AP.spa, verbose, Freq,
			     &CmdTools::savearray, (tmp)? &outpath_fname : NULL);
  }

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization and descaling Xr
  Xr *= AP.Ynrml;
  // msg.res *= Ynrml;

  Imr = BI->blob2multipixel(Xr, conf.dimObj);
  imr = Tools::multi_imsum(Imr);

  AP.nnz = Tools::l0norm(Xr);
  AP.sparsity = AP.nnz *1./ Xr.size();

  if (conf.phantom != "") {
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("Sparsity = %f\tUQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.sparsity, AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);
  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi");
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, *BI,  msg);
  if (bcsv_fname != "") {
    save_algo_parms_batch_csv(bcsv_fname, AP, *BI);
  }
  // Save reconstructed image(s)
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);
  if (AP.nbScale > 1)
    CmdTools::multi_imsave(Imr, buffer);

  vector<ImageXXd> Dimr = G->blob2pixelgrad(Xr, conf.dimObj);
  CmdTools::imsave(Dimr[0], outpath_fname+"/gradx");
  CmdTools::imsave(Dimr[1], outpath_fname+"/grady");
  
  // Change the output directory name if SNR is available
  if (conf.phantom != "") {
    sprintf(buffer, "%s.snr[%2.2f]", outpath_fname.c_str(), AP.snr);
    outpath_fname = CmdTools::rename_outpath(outpath_fname, buffer);
  }

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;
  
  if (verbose > 1) {
    CmdTools::imshow(imr, "Reconstructed blob image");
    if (AP.nbScale > 1)
      CmdTools::multi_imshow(Imr, "Reconstructed blob image");

    CmdTools::imshow(Dimr[0], "Reconstructed blob image gradX");
    CmdTools::imshow(Dimr[1], "Reconstructed blob image gradY");
  }

  return 0;
}

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[BlobImage Parameters]"<<endl;
  fout<<"box="<<AP.box<<endl;
  fout<<"roi="<<AP.roi<<endl;
  fout<<"fspls="<<AP.fspls<<endl;
  fout<<"cut_off_err="<<AP.cut_off_err<<endl;
  fout<<"fcut_off_err="<<AP.fcut_off_err<<endl;
  fout<<BI<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"stripint="<<AP.stripint<<endl;
  fout<<"Ynrml="<<AP.Ynrml<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"upSpl="<<AP.upSpl<<endl;
  fout<<"aniso="<<AP.aniso<<endl;
  fout<<"epsilon="<<AP.epsilon<<endl;
  fout<<"mu="<<AP.mu<<endl;
  fout<<"mu_rel="<<AP.mu_rel<<endl;
  fout<<"gamma="<<AP.gamma<<endl;
  fout<<"pgr="<<AP.penalty_grad<<endl;
  fout<<"pid="<<AP.penalty_id<<endl;
  fout<<endl;

  fout<<"tol="<<AP.tol<<endl;
  //fout<<"tolInn="<<AP.tolInn<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;
  fout<<"hIter="<<AP.hIter<<endl;
  fout<<"incr="<<AP.incr<<endl;
  fout<<"stol="<<AP.stol<<endl;
  fout<<"spa="<<AP.spa<<endl;
  fout<<"debias CG.="<<AP.cgmaxIter<<endl;
  fout<<"noapprx="<<AP.noapprx<<endl;

  // fout<<"rwIter="<<AP.rwIter<<endl;
  // fout<<"rwEpsilon="<<AP.rwEpsilon<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"sparsity="<<AP.sparsity<<endl;
  fout<<"nnz="<<AP.nnz<<endl;
  fout<<endl;

  fout<<"snr="<<AP.snr<<endl;
  fout<<"corr="<<AP.corr<<endl;
  fout<<"uqi="<<AP.uqi<<endl;
  fout<<"mse="<<AP.mse<<endl;
  fout<<"si="<<AP.si<<endl;
  fout<<endl;

  fout<<"time="<<AP.time<<endl;
  fout<<"niter="<<msg.niter<<endl;
  fout<<"normalized residual |Ax-b|="<<msg.res<<endl;
  fout<<"tvl1-norm="<<msg.norm<<endl;
  
  fout.close();
}


void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP, const BlobImage &BI)
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
    fp<<"BlobTVL1_batch_reconstruction_results"<<endl;
    // Acquisition
    fp<<"nbProj, "<<"pixDet, ";
    // BlobImage
    fp<<"fspls, "<<"nbNode, ";
    for (int n=0; n<BI.get_nbScale(); n++) {
	sprintf(buffer, "blob_radius_%d, bgrid_splStep_%d, ",n, n);
      fp<<buffer;
    }

    // Algo related
    fp<<"mu_rel, snr, corr, uqi, mse, si, time"<<endl;
  }

    //fp<<AP.hex<<", "<<AP.fspls<<", "<<AP.upSpl<<", "<<AP.cut_off_err<<", "<<AP.fcut_off_err<<", "<<AP.nbScale<<", "<<AP.dil<<", "<<AP.blob_name<<", ";
    fp<<AP.nbProj<<", "<<AP.pixDet<<", "<<AP.fspls<<", "<<BI.get_nbNode()<<", ";
  for (int n=0; n<BI.get_nbScale(); n++) {
    fp<<BI.blob[n]->radius<<", "<<BI.bgrid[n]->splStep<<", ";
  }
  //fp<<AP.stripint<<", "<<AP.tunning<<", ";
  fp<<AP.mu_rel<<", "<<AP.snr<<", "<<AP.corr<<", "<<AP.uqi<<", "<<AP.mse<<", "<<AP.si<<", "<<AP.time;
  fp<<endl;

  fp.close();  
}
