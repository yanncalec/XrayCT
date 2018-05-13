// Image reconstruction by TV minimization method and blob representation (with tunning of projector and autoscaling of parameter mu)

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg *msg);
void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP, const BlobImage &BI);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("2D Reconstruction by TV minimization on multiscale blob image using TVAL3 algorithm.\
Solve one of the following optimization problems:\n\
     min_f mu*|Af-b|^2 + TV(f), st. f>=0  (TV)\n\
     min_f TV(f), st. Af=b, and f>=0  (TVEQ)\n\
     min_f TV(f), st. |Af-b|<epsilon, and f>=0  (TVDN)\n\
Here TV(f) is the discrete TV norm of a blob image represented by blob coefficients f_1..f_N.\n\
About blob image:\n\
The blob image f(x) is a function of type:\n\
- Single scale: f(x)=sum_k f_k phi(x-x_k)\n\
- Multiscale: f(x) = sum_j sum_k f_{jk} phi_j(x-x_{jk})\n\
{f_k}, {f_{jk}} are blob coefficients, {phi(x-x_k)}, {phi(x-x_{jk})} are blobs centered at node x_k and x_{jk} (can be hexagonal or Cartesian lattice).\n\
Parameters of a blob image:\n\
Due to the localization character of blob, the reconstruction with a blob image is more stable (less high frequency error, faster convergence) than a pixel image of equivalent spatial resolution. \
Some important parameters of a blob image include:\n\
1. Number of scales. A multiscale blob image is composed of a coarse scale Gaussian blob image and several fine scale blob images of type X. The total number of scales is passed by --nbScale N (N>=1).\n\
2. Fine scale blob type X can be one of follows: Diff-Gauss, Mexican-hat, D4Gauss. (--blob diff, mex, d4gs)\n\
3. Dilation (>1) between scales, like the wavelet. (--dil N, N>1)\n\
The above parameters are unnecessary if the single scale (--nbScale 1) blob image is used: in this case the image is composed uniquely by Gaussian blob.\n\
4. Blob spatial radius and frequency radius. The Gaussian family blobs are not compactly supported, in practice they must be truncated both in spatial (--cut) and frequency (--fcut) domain. Truncation in frequency domain can change the density of blob grid.\n\
5. Reconstruction bandwidth, which is (partially) bounded by the detector bandwidth B, and can be adjusted by changing blob's parameter (dilation on blob). The parameter (--fspls N) changes the reconstruction bandwidth to B/N.\n\
For TV reconstruction:\n\
6. Gradient grid density (--upSpl N, N>=1) determines how well the discrete TV approximates the continuous TV norm. N is the up-sampling factor of the gradient grid wrt blob grid.\n\
For projector:\n\
7. On blob image the strip-integral projector can be easily evaluated (by passing --strip), nevertheless this is not recommended in general unless the detector pixel number is very low.\n\
Examples:\n\
    1. BlobTV.run DIR --fspls 4 --nbScale 1 --mu 0.1 --tol 1e-3 --iter 500 -vv\n\
solves the problem (TV) using a single scale Gaussian blob image on a hexagonal grid, with tolerance 1e-3 and 500 iterations. mu=0.1 is a factor of penalization(true value of mu is auto-scaled), the bandwidth of reconstruction is 1/4 of the optimal one.\n\
    2. BlobTV.run DIR --cat --fspls 2 --nbScale 1 --mu 0.1 --epsilon 0.1 --tol 1e-3 --iter 500 -vv\n\
solves the problem (TVDN) using a single scale Gaussian blob image on a Cartesian grid, the bandwidth of reconstruction is 1/2 of the optimal one.\n\
    3. BlobTV.run DIR --fspls 4 --nbScale 1 --mu 0.1 --rwIter 3 --rwEpsilon 1 --tol 1e-3 --iter 500 -vv\n\
solves the problem (TV) with 3 reweighted iterations.\n\
\nSuggestions on the choice of regularization parameter mu:\n\
Infortunately, there's no very scientific rules in setting mu, the most important parameter. We suggest the following manual method. Fix the acquisition configuration, the noise level and the blob image parameters, in command line run:\n\
        for MU in a0 a1...aN; do BlobTV.run --fspls X --mu $MU --bcsv test.csv ; done\n\
This will test a range of mu (in a0..aN) and write the reconstruction results (SNR etc) in a csv file test.csv. Choose manually the best mu then and use it for the current configuration.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for blob image and reconstruction. The hexagonal grid requires less samples (86.6%) than cartesian grid for representing a bandlimited function. [hexagonal grid]", cmd, false);

  TCLAP::ValueArg<double> roiArg("", "roi", "factor to effective reconstruction ROI. [0.8]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to reconstruction bandwidth. Bigger fspls, smaller reconstruction bandwidth : fspls=2 for a blob two times bigper and a grid 2 times coarser. \
Thumb rule : 4 for pixDet=1024, 1 for pixDet=256, etc. Use small value (<=0.5) for insufficient detector resolution and oversampled angular resolution. [2]", false, 2, "double", cmd);
  TCLAP::ValueArg<int> upSplArg("", "upSpl", "Up-sampling factor for gradient grid. In theory upSpl>1 increases the presicion of TV norm approximation (and the computation load), but in practice it has no visible effect for various grid density. [1]", false, 1, "int", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  //TCLAP::SwitchArg tunningSwitch("", "tunning", "projector is normalized by its spectral radius (st norm(A)<=1), and mu is intepreted as a factor [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-scale blob image model. nbScale=1 for gaussian blob image. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, meaningful only when nbScale>1. [2]", false, 2, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only when nbScale>1. Possible choices : 'mexhat', 'd4gs'. [mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain, this is a fine-tuning for grid, it increases the grid density without dilating the blob. Decrease this value when the grain or gibbs artifacts are observed. This can seriously increase the computation time. [1e-1]", false, 1e-1, "double", cmd);

  //TCLAP::SwitchArg tv2Switch("", "tv2", "using second order TV. [false]", cmd, false);
  TCLAP::SwitchArg anisoSwitch("", "aniso", "anisotropic TV model. [false]", cmd, false);
  TCLAP::SwitchArg nonnegSwitch("", "nonneg", "turn-on positive constraint. Positivity constraint may accelerate the convergence. It's automatically off if multiscale model is used. [false]", cmd, false);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "model selection: <0 for unconstraint TV, 0 for TVEQ, >0 for TVDN with epsilon as noise level. [-1]", false, -1, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "factor to the penalty coefficient for data fidelity. The most important and sensible parameter. Increasing mu gives more importance to data fidelity term. [0.5]", false, 0.5, "double", cmd);  
  TCLAP::ValueArg<double> betaArg("", "beta", "penalty coefficient for gradient penalty. Larger beta accelerate the convergence but reduce the precision. [100]", false, 100, "double", cmd);
  TCLAP::ValueArg<double> tolInnArg("", "tolInn", "TVAL3 : inner iterations stopping tolerance. Decrease for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "outer iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> rwEpsilonArg("", "rwEpsilon", "reweighting parameter between (0, 1], meaningful for rwIter>1. A grid node is treated as contour if its gradient norm is bigger than Cst*epsilon. Use small epsilon for strong edge detection/formation behavior. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

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
  AP.fspls = fsplsArg.getValue();
  AP.upSpl = upSplArg.getValue();

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

  AP.stripint = stripintSwitch.getValue();

  //bool tv2 = tv2Switch.getValue();
  AP.nonneg = (AP.nbScale>1) ? false : nonnegSwitch.getValue();
  AP.maxIter = IterArg.getValue();
  AP.tolInn = tolInnArg.getValue();
  AP.tol = tolArg.getValue();
  AP.mu = muArg.getValue();
  AP.beta = betaArg.getValue();
  AP.aniso = anisoSwitch.getValue();
  AP.epsilon = epsilonArg.getValue();

  AP.rwIter = rwIterArg.getValue();
  AP.rwEpsilon = rwEpsilonArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Create output path
  char buffer[256];
  sprintf(buffer, "BlobTV.at_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%1.1e]_tol[%1.1e]_epsilon[%1.1e]_rwIter[%d].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.mu, AP.tol, AP.epsilon, AP.rwIter, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, true, AP.roi);
  AP.nbProj = conf.nbProj_total;
  AP.pixDet = conf.pixDet;

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);  
  // Normalize the sinogram 
  double Ynrml = Y.abs().mean(); 
  Y /= Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its mean abs value: "<<Ynrml<<endl;
  // double sigma = Tools::sg_noiselevel(Y, conf.nbProj_total, conf.pixDet);
  // cout<<"Sinogram noise standard deviation : "<<sigma<<endl;

  // Pixel phantom image
  ImageXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  if (AP.blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, conf.diamFOV, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, conf.diamFOV, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err);
  }
  else if (AP.blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, conf.diamFOV, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.nbScale, AP.dil, AP.cut_off_err, AP.fcut_off_err);
  else {
    cerr<<"Unknown blob profile!"<<endl;
    exit(0);
  }
  
  Grid *igrid = new Grid(BI->bgrid[AP.nbScale-1]->sizeObj, BI->bgrid[AP.nbScale-1]->vshape*AP.upSpl, BI->bgrid[AP.nbScale-1]->splStep/AP.upSpl, AP.gtype, conf.diamFOV); // Interpolation grid

  // Init GPU Blob projector, normalize it such that |P|<=1
  BlobProjector *P;
  if (AP.stripint)
    P = new BlobProjector(conf, BI, 2);
  else
    P = new BlobProjector(conf, BI, 0);

  double norm_projector = P->estimate_opNorm();
  if (verbose)
    cout<<"Projector is normalized by the estimated norm: "<<norm_projector<<endl;    
  LinOp *M = new DiagOp(P->get_dimY(), 1./norm_projector);
  LinOp *A = new CompOp(M, P);

  // Gradient operator
  BlobInterpl *G = new BlobInterpl(BI, igrid, _Grad_);
  if (verbose > 2) {
    cout<<*P<<endl;
    cout<<*G<<endl;
  }

  // Normalization of mu
  AP.mu_rel = AP.mu;
  AP.mu *= igrid->nbNode; 

  // Initialization
  ArrayXd Xr = ArrayXd::Zero(P->get_dimX());

  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image

  ConvMsg* msg = new ConvMsg[AP.rwIter];
  clock_t t0 = clock();

  // The following is the reweighted iteration  
  {
    int gradRank = G->get_shapeY().x(); // dimension of gradient vector at each site
    //cout<<gradRank<<endl;
    ArrayXd W;			       // Reweighting coefficient
    if (AP.aniso)
      W.setOnes(G->get_dimY());
    else 
      W.setOnes(G->get_dimY() / gradRank);

    // Nu : initial value of lagrange multiplier for GX = U
    // Lambda : initial value of lagrange multiplier for AX = Y constraint
    ArrayXd Nu, Lambda;

    // double tolsc = 2.;
    // double tol_Inn0 = fmin(tol_Inn * pow(tolsc, rwIter-1), 1e-2);
    // double tol_Out0 = fmin(tol_Out * pow(tolsc, rwIter-1), 1e-2);

    for(int n=0; n<AP.rwIter; n++) {
      if (verbose)
	cout<<"\nReweighted TV minimization iteration : "<<n<<endl;    

      // Lambda and Nu should NOT be reused.
      Lambda.setZero(P->get_dimY());
      Nu.setZero(G->get_dimY());

      msg[n] = SpAlgo::TVAL3(*A, *G, W, Y,
			     Xr, Nu, Lambda, 
			     AP.epsilon, AP.aniso, AP.nonneg, AP.mu, AP.beta, 
			     AP.tolInn, AP.tol, AP.maxIter, verbose);
      // Benefit by refining reconstruction precision : better final result but the last iteration takes the most of time.
      //msg[n] = SpAlgo::TVAL3(A, G, W, Y, X, Nu, Lambda, eq, aniso, nonneg, mu, beta, tol_Inn0, tol_Out0, maxIter, verbose);

      // Update the edge weight and tolerations
      // tol_Inn0 /= tolsc;
      // tol_Out0 /= tolsc;

      // Update weighte
      if (AP.aniso)
	W = G->forward(Xr).abs();
      else
	W = SpAlgo::GTVNorm(G->forward(Xr), gradRank);
      double gmax = W.maxCoeff();
      
      ArrayXd Mask = (W <= AP.rwEpsilon * gmax / pow(2., n+1.)).select(1, ArrayXd::Zero(W.size())); // Edge mask
      ArrayXd toto(W.size());
      for (size_t m=0; m<W.size(); m++)
	toto[m] = 1/(W[m] + AP.rwEpsilon * gmax);
      W = toto  * Mask / toto.maxCoeff(); // This works the best
      // W = toto  * Mask; // No normalization 
      //W = toto  * Mask / toto.mean();      

      // Save weight and reconstruction images
      if (verbose > 1) {	      
	sprintf(buffer, "%s/rwIter_%d", outpath_fname.c_str(), n);
	//string rwIter_fname=buffer;
	ImageXXd imr = BI->blob2pixel(Xr, conf.dimObj);
	CmdTools::imsave(imr, buffer);

	sprintf(buffer, "%s/edge_rwIter_%d", outpath_fname.c_str(), n);
	//rwIter_fname = buffer;
	ImageXXd wg = BI->blob2pixel(W, conf.dimObj); // This wont work for MS blob. dimension of W is different of X
	CmdTools::imsave(wg, buffer);
      }
    }
  }

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization and descaling Xr
  Xr *= (Ynrml / norm_projector);  

  Imr = BI->blob2multipixel(Xr, conf.dimObj);
  imr = Tools::multi_imsum(Imr);
  if (conf.phantom != "") {
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
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
  fout<<BI<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"stripint="<<AP.stripint<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"upSpl="<<AP.upSpl<<endl;
  //  fout<<"tv2="<<AP.tv2<<endl;
  //fout<<"eq="<<AP.eq<<endl;
  fout<<"aniso="<<AP.aniso<<endl;
  fout<<"nonneg="<<AP.nonneg<<endl;
  fout<<"epsilon="<<AP.epsilon<<endl;
  fout<<"mu="<<AP.mu<<endl;
  fout<<"mu_rel="<<AP.mu_rel<<endl;
  fout<<"beta="<<AP.beta<<endl;

  fout<<"tol="<<AP.tol<<endl;
  fout<<"tolInn="<<AP.tolInn<<endl;
  fout<<"maxIter="<<AP.maxIter<<endl;

  fout<<"rwIter="<<AP.rwIter<<endl;
  fout<<"rwEpsilon="<<AP.rwEpsilon<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
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
    fout<<"tvnorm="<<msg[n].norm<<endl;
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
    fp<<"BlobTV_batch_reconstruction_results"<<endl;
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



  // cout<<"Blob grid :"<<endl;
  // cout<<*(BI->bgrid[nbScale-1])<<endl;
  // cout<<"Gradient grid :"<<endl;
  // cout<<*igrid<<endl;

  // // Test adjointness of projector
  // {
  //   ArrayXd X0, Y0, Z0;
  //   double err;
  //   for (int nn=0; nn<10; nn++) {
  //     X0 = ArrayXd::Random(P->get_dimX());
  //     Z0 = ArrayXd::Random(P->get_dimY());

  //     double v1=(Z0 * P->forward(X0)).sum();
  //     double v2=(X0 * P->backward(Z0)).sum();
  //     cout<<"<Z, P(X)> = "<<v1<<endl;
  //     cout<<"<Pt(Z), X> = "<<v2<<endl;
  //     err = v1-v2;
  //     printf("<Z, P(X)> - <Pt(Z), X> = %1.10e\n\n", err);
  //   }
  // }
