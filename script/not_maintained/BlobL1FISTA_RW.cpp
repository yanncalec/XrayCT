// Image reconstruction by L1 minimization method and blob representation
// This is a Reweighted L1 minimization (unconstraint optimization) using FISTA. Num exps show that on the unconstrainted optim problem, the RW scheme dosen't work, for the reason that the penalization constant \mu is 
// perturbed by the mask value. Use BlobL1PADM instead (constrainted optim) for solving the L1RW  problem.

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg *msg);
void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP, const BlobImage &BI);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by reweighted L1 minimization on multiscale blob image using FISTA.\
Solve:\n\
     min |Ax-b|^2 + mu*|x|_1\n\
A is the X-ray projector, and x is the blob coefficient.\
Examples:\n\
     BlobL1FISTA.run DIR --pix 512 --tol 1e-3 --iter 500 -vv\n\
solves the problem using haar wavelet on a Cartesian grid of dimension 512X512, tolerance 1e-3 and 500 iterations.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);
  TCLAP::ValueArg<string> xr_fnameArg("", "xr","reconstructed file name. Reuse the reconstructed coeffcient for reweighted l1 norm",false,"","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for reconstruction. [hexagonal grid]", cmd, false);
  TCLAP::ValueArg<double> roiArg("", "roi", "factor to effective reconstruction ROI. [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to ideal sampling step of reconstruction grid. [2]", false, 2, "double", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-blob image model. [4]", false, 4, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, fixed to 2 for diffgauss blob. [2.0]", false, 2.0, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only for nbScale > 1 : 'diff', 'mexhat', 'd4gs'.[mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. Reduce this value (lower than 1e-4) is not recommended for diff-gauss blob. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain. [1e-1]", false, 1e-1, "double", cmd);
  TCLAP::SwitchArg frameSwitch("", "frame", "use frame but not tight frame multiscale system [false]", cmd, false);
  TCLAP::SwitchArg besovSwitch("", "besov", "use Besov norm as initial reweighted l1-norm [false]", cmd, false);
  // TCLAP::ValueArg<double> nzArg("", "nz", "The percentage (between 0 and 1) of the biggest coeffcients. This defines a support set S and the algorithm exits when S is stablized. nz=0 or 1 turns off the constraint.  [0]", false, 0, "double", cmd);
  //TCLAP::SwitchArg debiasSwitch("", "debias", "debiasing on reconstruction using support information [false]", cmd, false);

  //TCLAP::ValueArg<string> algoArg("", "algo","L1-minimization algorithm : fista, ist. [fista]",false,"fista","string", cmd);

  //  TCLAP::ValueArg<double> muArg("", "mu", "normalized penalty coefficient for L1 fidelity. The default value 1e-5 works well for data of high SNR (~50 db). [1e-5]", false, 1e-5, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "normalized penalty coefficient for L1 fidelity. The default value works well for data of high SNR (~50 db). [2e-7]", false, 2e-7, "double", cmd);
  TCLAP::ValueArg<int> maxIterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> stolArg("", "stol", "support changes stopping tolerance. [5e-3]", false, 5e-3, "double", cmd);
  TCLAP::ValueArg<int> debiasArg("", "debias", "number cg iteration for debiasing. [25]", false, 25, "int", cmd);
  TCLAP::ValueArg<int> hIterArg("", "hIter", "number of homotopy continuation on mu. [4]", false, 4, "int", cmd);
  TCLAP::ValueArg<double> incrArg("", "incr", "homotopy continuation incremental factor. [0.5]", false, 0.5, "double", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> rwEpsilonArg("", "rwEpsilon", "reweighting parameter between (0, 1], meaningful for rwIter>1. Blob coefficients bigger than epsilon is treated as support, so small epsilon for strong support detection behavior. [1]", false, 1, "double", cmd);

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
  string xr_fname = xr_fnameArg.getValue();
  
  AlgoParms AP;			// Algo structure
  AP.hex = !catSwitch.getValue();
  AP.gtype = (AP.hex)?_Hexagonal_ : _Cartesian_;
  AP.roi = roiArg.getValue();
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
  // AP.nz = nzArg.getValue();
  // AP.debias = debiasSwitch.getValue();

  AP.stripint = stripintSwitch.getValue();
  AP.algo = "fista";

  AP.mu = muArg.getValue();
  AP.maxIter = maxIterArg.getValue();
  AP.tol = tolArg.getValue();
  AP.hIter = hIterArg.getValue();
  AP.incr = incrArg.getValue();
  AP.stol = stolArg.getValue();
  AP.cgmaxIter = debiasArg.getValue();

  AP.rwIter = rwIterArg.getValue();
  AP.rwEpsilon = rwEpsilonArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);
  
  // Create output path
  char buffer[256];
  sprintf(buffer, "BlobL1FISTA_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%1.1e]_tol[%1.1e]_rwIter[%d]_rwEpsilon[%2.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.mu, AP.tol, AP.rwIter, AP.rwEpsilon, pfix.c_str());  
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, true, AP.roi);
  AP.nbProj = conf.nbProj_total;
  AP.pixDet = conf.pixDet;

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);
  double Ynrml = Y.abs().mean(); // normalization factor to compensate the sinogram value impact
  Y /= Ynrml;

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

  if (verbose > 2) { 
    cout<<*BI<<endl;
    cout<<*P<<endl;
  }

  // Emperical value of optimal mu, it scales well for different data settings.
  AP.mu_rel = AP.mu;
  //double mu_factor = (1. * conf.pixDet * conf.nbProj_total) / AP.nbScale;
  double mu_factor = (conf.pixDet * conf.nbProj_total) * log(P->get_dimX());
  cout<<"Scaling mu by the factor: "<<mu_factor<<endl;
  AP.mu *= mu_factor;
  //AP.mu /= P->get_dimX();

  // Initialization
  ArrayXd Xr, Xr0;// = P->backward(Y);
  Xr.setZero(P->get_dimX());

  ArrayXd W = ArrayXd::Ones(Xr.size());			       // Reweighting coefficient
  ArrayXd toto(Xr.size());

  // If a data file is passed, use it as initialization and as reweighting coefficient.
  if (xr_fname != "") {
    cout<<"Load reconstructed blob coefficient..."<<endl;
    ArrayXd X0 = CmdTools::loadarray(xr_fname, BI->get_nbNode());
    toto = X0.abs();
    double gmax = toto.maxCoeff();
    for (size_t m=0; m<toto.size(); m++)
      toto[m] = 1/(toto[m] + AP.rwEpsilon * gmax);
    //W = toto / toto.maxCoeff();
    W = toto;
    W *= Xr.size() / W.sum();
    printf("W.sum=%f, W.min=%f, W.max=%f\n", W.sum(), W.minCoeff(), W.maxCoeff());    
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

  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image

  ConvMsg* msg = new ConvMsg[AP.rwIter];
  clock_t t0 = clock();

  // The following is the reweighted iteration  
  for(int n=0; n<AP.rwIter; n++) {
    if (verbose and AP.rwIter>1)
      cout<<"\nReweighted L1 minimization iteration : "<<n<<endl;    
    
    Xr.setZero();		// Clean first Xr
    //msg[n] = SpAlgo::L1FISTA(*P, W, Y, Xr, AP.mu, AP.tol, AP.maxIter, verbose); // replace nz here by (n+1)*AP.nz/AP.rwIter
    //msg[n] = SpAlgo::L1FISTA_Homotopy(*P, W, Y, Xr, AP.mu*Xr.size()/W.sum(), AP.tol, AP.maxIter, AP.stol, AP.hIter, AP.incr, AP.cgmaxIter, verbose);
    msg[n] = SpAlgo::L1FISTA_Homotopy(*P, W, Y, Xr, AP.mu, AP.tol, AP.maxIter, AP.stol, AP.hIter, AP.incr, AP.cgmaxIter, verbose);

    // update reweighting coefficeint
    W = Xr.abs();
    double gmax = W.maxCoeff();
    for (size_t m=0; m<W.size(); m++)
      W[m] = 1/(W[m] + AP.rwEpsilon * gmax);
    //printf("W.min=%f, W.max=%f\n", W.minCoeff(), W.maxCoeff());
    W /= W.maxCoeff();

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

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization of Xr
  Xr *= Ynrml;

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
  fout<<"fspls="<<AP.fspls<<endl;
  fout<<"cut_off_err="<<AP.cut_off_err<<endl;
  fout<<"fcut_off_err="<<AP.fcut_off_err<<endl;
  fout<<"tightframe="<<AP.tightframe<<endl;

  fout<<BI<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"stripint="<<AP.stripint<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"algo="<<AP.algo<<endl;
  fout<<"besov="<<AP.besov<<endl;
  fout<<"mu_rel="<<AP.mu_rel<<endl;
  fout<<"mu="<<AP.mu<<endl;
  fout<<"tol="<<AP.tol<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;

  fout<<"rwIter="<<AP.rwIter<<endl;
  fout<<"rwEpsilon="<<AP.rwEpsilon<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"nnz="<<AP.nnz<<endl;
  fout<<"sparsity="<<AP.sparsity<<endl;

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
    fp<<"mu_rel, nnz, snr, corr, uqi, mse, si, time"<<endl;
  }

  //fp<<AP.hex<<", "<<AP.fspls<<", "<<AP.upSpl<<", "<<AP.cut_off_err<<", "<<AP.fcut_off_err<<", "<<AP.nbScale<<", "<<AP.dil<<", "<<AP.blob_name<<", ";
  fp<<AP.algo<<", "<<AP.nbProj<<", "<<AP.pixDet<<", "<<AP.fspls<<", "<<BI.get_nbNode()<<", ";
  for (int n=0; n<BI.get_nbScale(); n++) {
    fp<<BI.blob[n]->radius<<", "<<BI.bgrid[n]->splStep<<", ";
  }
  //fp<<AP.stripint<<", "<<AP.tunning<<", ";
  fp<<AP.mu_rel<<", "<<AP.nnz<<", "<<AP.snr<<", "<<AP.corr<<", "<<AP.uqi<<", "<<AP.mse<<", "<<AP.si<<", "<<AP.time;
  fp<<endl;

  fp.close();  
}


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



// vector<ArrayXd> Q = BI->interscale_product(Xr);
// if (nterm_prct > 0 and nterm_prct < 1)
// 	nterm = min((int)ceil(nterm_prct * P->get_dimX()), nonzero);

// for (int m=0; m<Q.size(); m++) {
//  	printf("Scale %d, min = %e, max = %e\n", m, Q[m].minCoeff(), Q[m].maxCoeff());
// 	CmdTools::imshow(Q[m], BI->bgrid[m+2]->vshape.y(), BI->bgrid[m+2]->vshape.x(), "weight");	
// }

// W = BI->ProdThresh(Xr, epsilon, n);

// offset = 0;
// for (int j=0; j<BI->get_nbScale(); j++) {
// 	int N = BI->bgrid[j]->nbNode; // Number of coefficients (nodes) of scale j
// 	double xmax = Xr.segment(offset, N).abs().maxCoeff(); // Maximum magnitude of coefficients of scale j
// 	toto.setZero(N);
// 	for (size_t m=0; m<N; m++) {
// 	  toto[m] = 1/(fabs(X[offset + m]) + epsilon * xmax);
// 	}
// 	W.segment(offset, N) = toto / toto.maxCoeff() * pow(BI->get_scaling(), 0.1*j);
// 	//W.segment(offset, N) = toto * pow(BI->get_scaling(), 0.1*j);
// 	offset += N;
// }

// vector<ArrayXd> V = BI->separate(Xr);

// for (int m=0; m<V.size(); m++) {
//  	printf("Scale %d, min = %e, max = %e\n", m, V[m].minCoeff(), V[m].maxCoeff());
// 	CmdTools::imshow(V[m], BI->bgrid[m]->vshape.y(), BI->bgrid[m]->vshape.x(), "weight");	
// }
      
// double xmax = Xr.abs().maxCoeff(); // Maximum magnitude of coefficients of scale j
// for (size_t m=0; m<Xr.size(); m++)
// 	W[m] = 1/(fabs(Xr[m]) + epsilon * xmax);
// W = W / W.maxCoeff();
