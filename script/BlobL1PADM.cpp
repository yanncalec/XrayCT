// Image reconstruction by L1 minimization method (PADM) and blob representation

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg *msg);
void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP, const BlobImage &BI);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by reweighted L1 minimization on multiscale blob image.\n\
Solve:\n\
     min |x|_1 st Ax=b (BP)\n\
     min |x|_1 st |Ax-b|^2 < M*epsilon^2 (BPDN)\n\
A is the X-ray projector, and x is the blob coefficient. Examples:\n\
     BlobL1PADM.run DIR --blob mexhat --nbScale 4 --epsilon 5e-2 --tol 1e-4 --iter 1000 -v\n\
solves the problem using 4 scales Mexican hat blob system\n\
     BlobL1PADM.run DIR --blob mexhat --rwIter 6  --epsilon 5e-2 --tol 1e-4 --iter 1000 -vv\n\
solves the problem using 6 reweighted l1 minimization, and save the intermediate results.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);
  TCLAP::ValueArg<string> xr_fnameArg("", "xr","reconstructed file name. Reuse the reconstructed coeffcient for reweighted l1 norm",false,"","string", cmd);
  TCLAP::ValueArg<string> zr_fnameArg("", "zr","reconstructed lagrangian",false,"","string", cmd);
  TCLAP::ValueArg<string> rr_fnameArg("", "rr","reconstructed residual",false,"","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for reconstruction. [hexagonal grid]", cmd, false);

  TCLAP::ValueArg<double> roiArg("", "roi", "factor to effective reconstruction ROI. [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. By default(roi=1, box=1), the square ROI is the smallest square including circle ROI. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to ideal sampling step of reconstruction grid. [2]", false, 2, "double", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-blob image model. [4]", false, 4, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, fixed to 2 for diffgauss blob. [2.0]", false, 2.0, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only for nbScale > 1 : 'diff', 'mexhat', 'd4gs'.[mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. Reduce this value (lower than 1e-4) is not recommended for diff-gauss blob. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-1]", false, 1e-1, "double", cmd);
  TCLAP::SwitchArg frameSwitch("", "frame", "use frame but not tight frame multiscale system [false]", cmd, false);
  TCLAP::SwitchArg besovSwitch("", "besov", "use Besov norm as initial reweighted l1-norm [false]", cmd, false);

  TCLAP::ValueArg<double> betaArg("", "beta", "PADM : normalized data fitnesss penalty, small beta for large shrinkage effect. [7.5e3]", false, 7.5e3, "double", cmd);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "PADM : model selection: 0 for BPEQ, >0 for BPDN with epsilon as noise level per detector, i.e. |Ax-b|^2<M*epsilon^2 with M the sinogram dimension. [Use BlobCG.run to estimate this value then double it]", false, 5e-2, "double", cmd);
  TCLAP::ValueArg<double> tauArg("", "tau", "PADM : proximal penalty. [1.]", false, 1., "double", cmd);
  TCLAP::ValueArg<double> gammaArg("", "gamma", "PADM : lagrangian step. [1.]", false, 1., "double", cmd);

  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> maxIterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> rwEpsilonArg("", "rwEpsilon", "reweighting parameter between (0, 1], meaningful for rwIter>1. Blob coefficients bigger than epsilon are treated as support, so small epsilon for strong support detection behavior. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);
  TCLAP::ValueArg<int> FreqArg("", "mfreq", "print convergence message every mfreq iterations. [50]", false, 50, "int", cmd);

  cmd.parse(argc, argv);

  AlgoParms AP;			// Algo structure
  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string bcsv_fname = bcsv_fnameArg.getValue();
  string pfix = pfixArg.getValue();
  if (pfix == "")
    pfix = getenv ("SYSNAME");

  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();
  string xr_fname = xr_fnameArg.getValue();
  string zr_fname = zr_fnameArg.getValue();
  string rr_fname = rr_fnameArg.getValue();
  
  AP.hex = !catSwitch.getValue();
  AP.gtype = (AP.hex)?_Hexagonal_ : _Cartesian_;
  AP.roi = roiArg.getValue();
  AP.box = boxArg.getValue();

  AP.sgNrml = !nosgNrmlSwitch.getValue();
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
  }

  AP.cut_off_err = cut_off_errArg.getValue();
  AP.fcut_off_err = fcut_off_errArg.getValue();
  AP.tightframe = !frameSwitch.getValue();
  AP.besov = besovSwitch.getValue();

  AP.stripint = stripintSwitch.getValue();
  AP.algo = "padm"; //algoArg.getValue();

  AP.beta = betaArg.getValue();
  AP.epsilon = epsilonArg.getValue();
  AP.tau = tauArg.getValue();
  AP.gamma = gammaArg.getValue();
  AP.maxIter = maxIterArg.getValue();
  AP.tol = tolArg.getValue();

  AP.rwIter = rwIterArg.getValue();
  AP.rwEpsilon = rwEpsilonArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();
  int Freq = FreqArg.getValue();

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
  else if (AP.blobname == "diff") {
    BI = BlobImageTools::MultiGaussDiffGauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  }
  else if (AP.blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  }
  else if (AP.blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err, AP.tightframe);
  else {
    cerr<<"Unknown blob profile!"<<endl;
    exit(0);
  }
  // Guarantee that the blob image nodes are all inside FOV
  int err = BlobImageTools::BlobImage_FOV_check(*BI, conf.diamFOV);
  if (err>=0) {
    printf("Warning: blob image active nodes exceed FOV(diamFOV=%f)\n",conf.diamFOV);
    printf("Scale %d: diamROI=%f, sizeObj.x=%f, sizeObj.y=%f\n", err, BI->bgrid[err]->diamROI, BI->bgrid[err]->sizeObj.x(), BI->bgrid[err]->sizeObj.y()); 
    //exit(0);
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

  // Normalize the projector
  if (verbose)
    cout<<"Estimation of projector norm..."<<endl;
  AP.norm_projector = P->estimate_opNorm();
  if (verbose)
    cout<<"Projector is normalized by the estimated norm: "<<AP.norm_projector<<endl;
    
  LinOp *M = new DiagOp(P->get_dimY(), 1./AP.norm_projector);
  LinOp *A = new CompOp(M, P);

  if (verbose > 2) { 
    cout<<*BI<<endl;
    cout<<*P<<endl;
  }

  AP.beta_rel = AP.beta;
  double beta_factor = AP.norm_projector / (conf.pixDet * conf.nbProj_total * log(P->get_dimX())); 
  AP.beta *= beta_factor;
  if (verbose)
    cout<<"Scaling beta by the factor: "<<beta_factor<<endl;

  //AP.beta *= 2 * Y.size() / Y.abs().sum();
  //AP.beta *= sqrt(P->get_dimX());

  // Create output path
  char buffer[256];
  sprintf(buffer, "BlobL1PADM_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_epsilon[%1.1e]_beta[%1.1e]_tol[%1.1e]_rwIter[%d]_rwEpsilon[%2.2f]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.epsilon, AP.beta_rel, AP.tol, AP.rwIter, AP.rwEpsilon, AP.roi, AP.box, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // Initialization
  ArrayXd Xr, Xr0;// = P->backward(Y);
  ArrayXd Z;			// Lagrangian multiplier
  ArrayXd R;			// Residual or the auxilary variable
  Xr.setZero(P->get_dimX());

  ArrayXd W = ArrayXd::Ones(Xr.size());	// W is the reweighting coefficient
  ArrayXd toto(Xr.size());

  // If a data file is passed, use it as initialization and as reweighting coefficient.
  if (xr_fname != "") {
    cout<<"Load reconstructed blob coefficient..."<<endl;
    Xr = CmdTools::loadarray(xr_fname, BI->get_nbNode());
    W = BI->reweighting_mask(Xr, AP.rwEpsilon);
    // toto = Xr.abs();
    // double gmax = toto.maxCoeff();
    // for (size_t m=0; m<toto.size(); m++)
    //   toto[m] = 1/(toto[m] + AP.rwEpsilon * gmax);
    // W = toto / toto.maxCoeff();
    //printf("W.min=%f, W.max=%f\n", W.minCoeff(), W.maxCoeff());        
  }
  else if (AP.besov) {
    vector<ArrayXd> vW = BI->separate(W);
    double fc = 1;
    cout<<"MS blob scaling : "<<BI->get_scaling()<<endl;
    
    for (int s=0; s<vW.size(); s++) {
      vW[s] *= fc;
      fc *= BI->get_scaling();
    }
    W = BI->joint(vW);		// Scale dependant reweighting
  }

  if (zr_fname != "" and rr_fname != "") {
    Z = CmdTools::loadarray(zr_fname, Y.size());
    R = CmdTools::loadarray(rr_fname, Y.size());
  }
  else{
    Z.setZero(Y.size()); 
    R.setZero(Y.size()); 
  }

  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image

  ConvMsg* msg = new ConvMsg[AP.rwIter];
  clock_t t0 = clock();

  // The following is the reweighted iteration  

  for(int n=0; n<AP.rwIter; n++) {
    if (verbose)
      cout<<"\nReweighted L1 minimization iteration : "<<n<<endl;    

    if (AP.algo == "padm") {
      // msg[n] = SpAlgo::L1PADM_Homotopy(*A, *P, W, Y, Xr, 
      // 				       AP.epsilon*sqrt(conf.nbProj_total*conf.pixDet), // Noise level epsilon is scaled
      // 				       AP.beta, AP.tau, AP.gamma,
      // 				       AP.tol, AP.maxIter, AP.spa, 
      // 				       AP.stol, AP.hIter, AP.incr, AP.cgmaxIter, verbose);

      // Use last iteration result as initialization
      // Xr.setZero(); Z.setZero(); R.setZero();
      msg[n] = SpAlgo::L1PADM(*A, W, Y, Xr, Z, R,
      			      AP.epsilon*sqrt(conf.nbProj_total*conf.pixDet), // Noise level epsilon is scaled
      			      0, AP.beta, AP.tau, AP.gamma, 
      			      AP.tol, AP.maxIter, 0, 0, verbose, Freq);
    }
    
    // update reweighting coefficeint
    W = BI->reweighting_mask(Xr, AP.rwEpsilon);

    if (verbose > 1) {
      sprintf(buffer, "%s/rwIter_%d", outpath_fname.c_str(), n);
      ImageXXd imr = BI->blob2pixel(Xr, conf.dimObj);
      CmdTools::imsave(imr, buffer);

      vector<ImageXXd> Imr = BI->blob2multipixel(Xr, conf.dimObj);
      //vector<ImageXXd> Imr = BI->blob2multipixel(W, conf.dimObj);
      CmdTools::multi_imsave(Imr, buffer);
    }
  }

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization of Xr
  Xr *= (AP.Ynrml / AP.norm_projector);
  //Xr /= AP.norm_projector;

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
  sprintf(buffer, "%s/zr", outpath_fname.c_str());
  CmdTools::savearray(Z, buffer);
  sprintf(buffer, "%s/rr", outpath_fname.c_str());
  CmdTools::savearray(R, buffer);

  // if (AP.debias) {
  //   sprintf(buffer, "%s/xr0", outpath_fname.c_str()); // non debiased data
  //   CmdTools::savearray(Xr0, buffer);
  // }
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
  }

  return 0;
}


void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg *msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[BlobImage Parameters]"<<endl;
  fout<<"Ratio of squared ROI to squared FOV (no incidence in simulation mode) (box)="<<AP.box<<endl;
  fout<<"Ratio of ROI to FOV (no incidence in simulation mode) (roi)="<<AP.roi<<endl;
  fout<<"Factor to the blob lattice sampling step (fspls)="<<AP.fspls<<endl;
  fout<<"Blob spatial cut-off error (cut_off_err)="<<AP.cut_off_err<<endl;
  fout<<"Blob frequency cut-off error (fcut_off_err)="<<AP.fcut_off_err<<endl;
  fout<<"Use tight frame system (tightframe)="<<AP.tightframe<<endl;
  fout<<BI<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"Use strip-integral projector (stripint)="<<AP.stripint<<endl;
  fout<<"Estimated projector 2-norm (norm_projector)="<<AP.norm_projector<<endl;
  fout<<"Sinogram normalization value (Ynrml)="<<AP.Ynrml<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"Name of algorithm="<<AP.algo<<endl;
  fout<<"Use scale dependant Besov norm (besov)="<<AP.besov<<endl;
  fout<<"Noise level per detector bin (epsilon)="<<AP.epsilon<<endl;
  fout<<"PADM penalization (beta_rel)="<<AP.beta_rel<<endl;
  fout<<"PADM scaled penalization (beta)="<<AP.beta<<endl;
  fout<<"PADM lagrangian update parameter (gamma)="<<AP.gamma<<endl;
  fout<<"PADM proximity parameter (tau)="<<AP.tau<<endl;
  fout<<endl;

  fout<<"Global stopping tolerance (tol)="<<AP.tol<<endl;
  fout<<"Maximum number of iterations (maxIter)="<<AP.maxIter<<endl;

  fout<<"L1 reweighting iteration number (rwIter)="<<AP.rwIter<<endl;
  fout<<"L1 reweighting support detection sensibility (rwEpsilon)="<<AP.rwEpsilon<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"Number of non zero blobs (nnz)="<<AP.nnz<<endl;
  fout<<"Percentage of non zero blobs (sparsity)="<<AP.sparsity<<endl;
  fout<<endl;

  fout<<"SNR="<<AP.snr<<endl;
  fout<<"Streak Index (si)="<<AP.si<<endl;
  fout<<"CORR="<<AP.corr<<endl;
  fout<<"QI="<<AP.uqi<<endl;
  fout<<"MSE="<<AP.mse<<endl;
  fout<<"Time="<<AP.time<<endl;
  fout<<endl;

  for (int n=0; n<AP.rwIter; n++) {
    fout<<endl<<"Reweighted iteration : "<<n<<endl;
    fout<<"Number of iterations="<<msg[n].niter<<endl;
    fout<<"Normalized residual |Ax-b| ="<<msg[n].res<<endl;
    fout<<"L1 norm="<<msg[n].norm<<endl;
  }

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
    fp<<"BlobL1_batch_reconstruction_results"<<endl;

    // BlobImage
    fp<<"algo, "<<"nbProj, "<<"pixDet, "<<"fspls, "<<"nbNode, ";
    for (int n=0; n<BI.get_nbScale(); n++) {
      sprintf(buffer, "blob_radius_%d, bgrid_splStep_%d, ",n, n);
      fp<<buffer;
    }

    // Algo related
    fp<<"beta_rel, beta, epsilon, nnz, snr, corr, uqi, mse, si, time"<<endl;
  }

  //fp<<AP.hex<<", "<<AP.fspls<<", "<<AP.upSpl<<", "<<AP.cut_off_err<<", "<<AP.fcut_off_err<<", "<<AP.nbScale<<", "<<AP.dil<<", "<<AP.blob_name<<", ";
  fp<<AP.algo<<", "<<AP.nbProj<<", "<<AP.pixDet<<", "<<AP.fspls<<", "<<BI.get_nbNode()<<", ";
  for (int n=0; n<BI.get_nbScale(); n++) {
    fp<<BI.blob[n]->radius<<", "<<BI.bgrid[n]->splStep<<", ";
  }
  //fp<<AP.stripint<<", "<<AP.tunning<<", ";
  fp<<AP.beta_rel<<", "<<AP.beta<<", "<<AP.epsilon<<", "<<AP.nnz<<", "<<AP.snr<<", "<<AP.corr<<", "<<AP.uqi<<", "<<AP.mse<<", "<<AP.si<<", "<<AP.time;
  fp<<endl;

  fp.close();  
}
