// Image reconstruction by TV minimization method and blob representation

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, double fspls, int upSpl, const BlobImage &BI,
		     double cut_off_err, double fcut_off_err, bool stripint, 
		     bool eq, bool aniso, bool nonneg, double mu, double beta,
		     double tol, int maxIter, int rwIter, double epsilon, 
		     double snr, double corr, double time, const ConvMsg *msg);

int main(int argc, char **argv) {

  TCLAP::CmdLine cmd("", ' ', "0.1");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for blob image and reconstruction. The hexagonal grid requires less samples (86.6%) than cartesian grid for representing a bandlimited function. [hexagonal grid]", cmd, false);

  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to reconstruction bandwidth. Bigger fspls, smaller reconstruction bandwidth : fspls=2 for a blob two times big and a grid 2 times coarse. Thumb rule : 4 for pixDet=1024, 1 for pixDet=256, etc. Particular case : fspls = 0.5 for pixDet=128 (insufficient detector resolution) and nbProj=128 (oversampled angular resolution). [4]", false, 4, "double", cmd);
  TCLAP::ValueArg<int> upSplArg("", "upSpl", "Up-sampling factor for interpolation grid. In theory upSpl>1 increases the presicion of TV norm evaluation (and the computation load), but in practice it has no visible effect for various grid density. [1]", false, 1, "int", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. The strip-integral is a more [false]", cmd, false);
  TCLAP::SwitchArg notunningSwitch("", "notunning", "no normalization for projector [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-scale blob image model. nbScale=1 for gaussian blob image. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, meaningful only when nbScale>1. [1.5]", false, 1.5, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only when nbScale>1. Possible choices : 'mexhat', 'd4gs'. [mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain, this is a fine-tuning for grid, it increases the grid density without dilating the blob. Decrease this value when the grain or gibbs artifacts are observed. This can seriously increase the computation time. [1e-1]", false, 1e-1, "double", cmd);

  //TCLAP::SwitchArg tv2Switch("", "tv2", "using second order TV. [false]", cmd, false);
  TCLAP::SwitchArg eqSwitch("", "eq", "Equality constraint model. Turn this on in high SNR sinogram (> 50 db) can greatly boost reconstruction quality, but increase also the computation time. [false]", cmd, false);
  TCLAP::SwitchArg anisoSwitch("", "aniso", "anisotropic TV model. [false]", cmd, false);
  TCLAP::SwitchArg noposSwitch("", "nopos", "turn-off positive constraint. Positivity constraint accelerates the convergence. It's automatically off if multiscale model is used. [false]", cmd, false);
  TCLAP::ValueArg<double> nzArg("", "nz", "l0 constraint. The percentage (between 0 and 1) of the biggest coeffcients to be kept. 0 or 1 turns off the constraint.  [0]", false, 0, "double", cmd);

  TCLAP::ValueArg<double> muArg("", "mu", "factor to the penalty coefficient for data fidelity. The most important and sensible parameter. Increasing mu gives more importance to data fidelity term. [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> betaArg("", "beta", "penalty coefficient for gradient penalty. Larger beta accelerate the convergence but reduce the precision. [10]", false, 10, "double", cmd);
  //TCLAP::ValueArg<double> tolInnArg("", "tolInn", "inner iterations stopping tolerance. Decrease for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "outer iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "reweighting parameter between (0, 1], meaningful for rwIter>1. Grid node of gradient norm bigger than epsilon is treated as contour, so small epsilon for strong edge detection/formation behavior. [1]", false, 1, "double", cmd);
  //TCLAP::SwitchArg edgeSwitch("", "edge", "use edge detection other than edge reweighting in reweighted iteration, meaningful for rwIter>1. [false]", cmd, false);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);

  cmd.parse(argc, argv);

  string inpath_fname = inpath_fnameArg.getValue();
  string outpath_fname = outpath_fnameArg.getValue();
  string cfg_fname = inpath_fname + "/" + cfg_fnameArg.getValue();
  string sg_fname = inpath_fname + "/" + sg_fnameArg.getValue();

  bool hex = !catSwitch.getValue();
  GridType gtype = (hex)?_Hexagonal_ : _Cartesian_;
  double fspls = fsplsArg.getValue();
  int upSpl = upSplArg.getValue();

  int nbScale = nbScaleArg.getValue(); 
  string blobname = blobnameArg.getValue();
  double dil =  dilArg.getValue();

  if (nbScale == 1) {
    blobname = "gauss";
    dil = 1;
  }
  else {
    assert(blobname == "mexhat" || blobname == "d4gs");
    assert(dil > 1);
  }

  double cut_off_err = cut_off_errArg.getValue();
  double fcut_off_err = fcut_off_errArg.getValue();

  bool stripint = stripintSwitch.getValue();
  bool tunning = !notunningSwitch.getValue();

  //bool tv2 = tv2Switch.getValue();
  //bool nonneg = !noposSwitch.getValue() and (nbScale == 1);
  bool nonneg = !noposSwitch.getValue();
  int maxIter = IterArg.getValue();
  //double tol_Inn = tolInnArg.getValue();
  double tol = tolArg.getValue();
  double mu = muArg.getValue();
  double beta = betaArg.getValue();
  double nz = nzArg.getValue();
  bool aniso = anisoSwitch.getValue();
  bool eq = eqSwitch.getValue();

  int rwIter = rwIterArg.getValue();
  double epsilon = epsilonArg.getValue();
  //  bool edge = edgeSwitch.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Create output path
  char buffer[256];
  sprintf(buffer, "BlobTV_fspls[%2.2f]_fcut[%2.1f]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%2.1f]_beta[%2.1f]_nz[%2.2f]_tol[%1.1e]_eq[%d]_rwIter[%d]_epsilon[%2.1f]", fspls, fcut_off_err, nbScale, dil, blobname.c_str(), mu, beta, nz, tol, eq, rwIter, epsilon);
  //sprintf(buffer, "BlobTV_fspls[%2.2f]_fcut[%2.1f]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%1.1e]_beta[%1.1e]_nz[]_tol[%1.1e]_eq[%d]_rwIter[%d]_epsilon[%1.1e]", fspls, fcut_off_err, nbScale, dil, blobname.c_str(), mu, beta, tol, eq, rwIter, epsilon);
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);  
  // normalize first Y by its maximum coefficient
  double Ynrml = Y.abs().maxCoeff(); // normalization factor to compensate the sinogram value impact
  Y /= Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its maximum coefficient : "<<Ynrml<<endl;
  // if (Ynrml > 1) {
  //   Y /= Ynrml;
  //   if (verbose)
  //     cout<<"Sinogram is normalized by its maximum coefficient : "<<Ynrml<<endl;
  // }

  // double sigma = Tools::sg_noiselevel(Y, conf.nbProj_total, conf.pixDet);
  // cout<<"Sinogram noise standard deviation : "<<sigma<<endl;

  // Pixel phantom image
  ArrayXXd im;
  if (conf.phantom != "")
    im = CmdTools::imread(conf.phantom.data());

  BlobImage *BI;
  if (blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, cut_off_err, fcut_off_err);
  }
  else if (blobname == "mexhat") {
    BI = BlobImageTools::MultiGaussMexHat(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, dil, cut_off_err, fcut_off_err);
    //mu *= 20;		
  }
  else if (blobname == "d4gs")
    BI = BlobImageTools::MultiGaussD4Gauss(conf.sizeObj, gtype, 1./Acq::nrml_spDet(conf)/fspls, nbScale, dil, cut_off_err, fcut_off_err);
  else {
    cerr<<"Unknown blob profile!"<<endl;
    exit(0);
  }
  
  Grid *igrid = new Grid(BI->bgrid[nbScale-1]->sizeObj, BI->bgrid[nbScale-1]->vshape*upSpl, BI->bgrid[nbScale-1]->splStep/upSpl, gtype); // Interpolation grid

  // Init GPU Blob projector
  BlobProjector *P;
  if (stripint) {
    P = new BlobProjector(conf, BI, 2);
  }
  else
    P = new BlobProjector(conf, BI, 0);
  if (tunning)			// Normalize the projector
    P->set_precond();

  // Gradient operator
  BlobInterpl *G = new BlobInterpl(BI, igrid, _Grad_);
  if (verbose >1) {
    cout<<*P<<endl;
    cout<<*G<<endl;
  }

  // l0 constraint
  int nterm = 0;
  if (nz > 0 and nz < 1) {
    nterm = (int)ceil(nz * P->get_dimX());
  }

  // Normalization of mu
  mu *= 2.5e-5 * (1.*conf.pixDet*conf.nbProj_total) / igrid->nbNode; 
  // mu *= sqrt(igrid->nbNode) / (conf.pixDet*conf.nbProj_total); // this gives mu ~ 2e7
  // Explenation : the data fittness (noise) energy err (|Ax-b|<= err) scales roughly like err \propto conf.pixDet*conf.nbProj_total
  // The TV energy scales like |x|_TV \propto igrid->nbNode

  // Initialization
  //ArrayXd Xr = P->backward(Y);
  ArrayXd Xr = ArrayXd::Zero(P->get_dimX());

  double recsnr = 0;		// SNR and Corr of reconstruction
  double reccorr = 0;
  vector<ArrayXXd> Imr;		// Reconstructed multiscale image
  ArrayXXd imr;			// Sum of reconstructed image

  ConvMsg* msg = new ConvMsg[rwIter];
  clock_t t0 = clock();
  // The following is the reweighted iteration
  
  {
    int gradRank = G->get_shapeY().x(); // dimension of gradient vector at each site

    ArrayXd W;			       // Reweighting coefficient
    if (aniso)
      W.setOnes(G->get_dimY());
    else 
      W.setOnes(G->get_dimY() / gradRank);

    // Nu : initial value of lagrange multiplier for GX = U
    // Lambda : initial value of lagrange multiplier for AX = Y constraint
    ArrayXd Nu, Lambda;

    // double tolsc = 2.;
    // double tol_Inn0 = fmin(tol_Inn * pow(tolsc, rwIter-1), 1e-2);
    // double tol_Out0 = fmin(tol_Out * pow(tolsc, rwIter-1), 1e-2);

    for(int n=0; n<rwIter; n++) {
      if (verbose)
	cout<<"\nReweighted TV minimization iteration : "<<n<<endl;    

      // Lambda and Nu should NOT be reused.
      Lambda.setZero(P->get_dimY());
      Nu.setZero(G->get_dimY());

      msg[n] = SpAlgo::TVL0AL3(*P, *G, W, Y,
			       Xr, Nu, Lambda, 
			       eq, aniso, nonneg, nterm,
			       mu, beta, 
			       1e-4, tol, maxIter, verbose);

      // ArrayXi Idx = (Xr.abs()>0).select(1, ArrayXi::Zero(Xr.size()));
      // cout<<"Non zero terms :"<<Idx.sum()<<endl;
      // cout<<"Percentage : "<<Idx.sum()*1./Idx.size()<<endl;

      // Benefit by refining reconstruction precision : better final result but the last iteration takes the most of time.
      //msg[n] = SpAlgo::TVAL3(A, G, W, Y, X, Nu, Lambda, eq, aniso, nonneg, mu, beta, tol_Inn0, tol_Out0, maxIter, verbose);

      // Update the edge weight and tolerations
      // tol_Inn0 /= tolsc;
      // tol_Out0 /= tolsc;

      if (aniso)
	W = G->forward(Xr).abs();
      else
	W = SpAlgo::GTVNorm(G->forward(Xr), gradRank);
      double gmax = W.maxCoeff();
      
      ArrayXd Mask = (W <= epsilon * gmax / pow(2., n+1.)).select(1, ArrayXd::Zero(W.size())); // Edge mask
      ArrayXd toto(W.size());
      for (size_t m=0; m<W.size(); m++)
	toto[m] = 1/(W[m] + epsilon * gmax);
      W = toto  * Mask / toto.maxCoeff(); // This works the best
      // W = toto  * Mask; // No normalization 
      //W = toto  * Mask / toto.mean();      

      if (verbose > 1) {	      
	sprintf(buffer, "%s/rwIter_%d", outpath_fname.c_str(), n);
	//string rwIter_fname=buffer;
	ArrayXXd imr = BI->blob2pixel(Xr, conf.dimObj);
	CmdTools::imsave(imr, buffer);

	sprintf(buffer, "%s/edge_rwIter_%d", outpath_fname.c_str(), n);
	//rwIter_fname = buffer;
	ArrayXXd wg = BI->blob2pixel(W, conf.dimObj); // This wont work for MS blob. dimension of W is different of X
	CmdTools::imsave(wg, buffer);
      }
    }
  }

  double rectime = (clock()-t0)/(double)CLOCKS_PER_SEC;

  //Xr *= Ynrml;			// Denormalization
  printf("Reconstruction taken %lf seconds\n", rectime); 

  Imr = BI->blob2multipixel(Xr, conf.dimObj);
  imr = Tools::multi_imsum(Imr);
  if (conf.phantom != "") {
    recsnr = Tools::SNR(imr, im);
    reccorr = Tools::Corr(imr, im);
    printf("Corr. = %f\t, SNR = %f\n", reccorr, recsnr);
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer);
  // Save BlobImage object
  BlobImageTools::save(*BI, outpath_fname+"/bi");
  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", fspls, upSpl, *BI,
		  cut_off_err, fcut_off_err, stripint,
		  eq, aniso, nonneg, mu, beta, 
		  tol, maxIter, rwIter, epsilon, 
		  recsnr, reccorr, rectime, msg);

  // Save reconstructed image(s)
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);
  if (nbScale > 1)
    CmdTools::multi_imsave(Imr, buffer);

  vector<ArrayXXd> Dimr = G->blob2pixelgrad(Xr, conf.dimObj);
  CmdTools::imsave(Dimr[0], outpath_fname+"/gradx");
  CmdTools::imsave(Dimr[1], outpath_fname+"/grady");
  
  cout<<"Outputs saved in directory "<<outpath_fname<<endl;
  
  if (verbose) {
    CmdTools::imshow(imr, "Reconstructed blob image");
    if (nbScale > 1)
      CmdTools::multi_imshow(Imr, "Reconstructed blob image");

    CmdTools::imshow(Dimr[0], "Reconstructed blob image gradX");
    CmdTools::imshow(Dimr[1], "Reconstructed blob image gradY");
  }

  return 0;
}

void save_algo_parms(const string &fname, double fspls, int upSpl,const BlobImage &BI,
		     double cut_off_err, double fcut_off_err, bool stripint, 
		     bool eq, bool aniso, bool nonneg, double mu, double beta,
		     double tol, int maxIter, int rwIter, double epsilon, 
		     double snr, double corr, double time, const ConvMsg *msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[BlobImage Parameters]"<<endl;
  fout<<"fspls="<<fspls<<endl;
  fout<<"cut_off_err="<<cut_off_err<<endl;
  fout<<"fcut_off_err="<<fcut_off_err<<endl;
  fout<<BI<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"raytracing="<<!stripint<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"upSpl="<<upSpl<<endl;
  //  fout<<"tv2="<<tv2<<endl;
  fout<<"eq="<<eq<<endl;
  fout<<"aniso="<<aniso<<endl;
  fout<<"nonneg="<<nonneg<<endl;
  fout<<"mu="<<mu<<endl;
  fout<<"beta="<<beta<<endl;

  fout<<"tol="<<tol<<endl;
  fout<<"maxIter="<<maxIter<<endl;

  fout<<"rwIter="<<rwIter<<endl;
  fout<<"epsilon="<<epsilon<<endl;
  fout<<endl;

  fout<<"[Results]"<<endl;
  fout<<"snr="<<snr<<endl;
  fout<<"corr="<<corr<<endl;
  fout<<"time="<<time<<endl;

  for (int n=0; n<rwIter; n++) {
    fout<<endl<<"Reweighted iteration : "<<n<<endl;
    fout<<"niter="<<msg[n].niter<<endl;
    fout<<"res="<<msg[n].res<<endl;
    fout<<"tvnorm="<<msg[n].norm<<endl;
  }

  fout.close();
}

  // cout<<"Blob grid :"<<endl;
  // cout<<*(BI->bgrid[nbScale-1])<<endl;
  // cout<<"Gradient grid :"<<endl;
  // cout<<*igrid<<endl;
