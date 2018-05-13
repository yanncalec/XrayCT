// Image reconstruction by TV minimization method and pixel representation (with tunning and autoscaling)
#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg *msg);
void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP);

int main(int argc, char **argv) {

   TCLAP::CmdLine cmd("2D Reconstruction by TV minimization using TVAL3 method on pixel \
image. Solve one of the following optimization problems:\n\
      min_x mu*|Ax-b|^2 + TV(x), st. x>=0  (TV)\n\
      min_x TV(x), st. Ax=b, and x>=0  (TVEQ)\n\
      min_x TV(x), st. |Ax-b|<epsilon, and x>=0  (TVDN)\n\
Examples:\n\
1. PixTV.run DIR --pix 256 --mu 5e-2 --tol 1e-3 --iter 500 -vv\n\
    solves the problem (TV), on a image of dimension 256^2, with tolerance 1e-3 and 500 iterations.\n\
2. PixTV.run DIR --pix 512 --mu 1e-2 --epsilon 10 --rwIter 4 --rwEpsilon 0.1 -vv\n\
    solves the problem (TVDN, epsilon=10) using 4 reweighted multistep iterations, on a image of dimension 512^2 with tolerance 1e-4, the intermediate results are saved.");
   
  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::ValueArg<int> pixArg("", "pix", "reconstruction dimension of pixel image (larger side). [256]", false, 256, "int", cmd);
  TCLAP::SwitchArg anisoSwitch("", "aniso", "anisotropic TV model. [false]", cmd, false);
  TCLAP::SwitchArg nonnegSwitch("", "nonneg", "turn-on positive constraint. Positivity constraint may accelerate the convergence. [false]", cmd, false);

  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "model selection: <0 for unconstraint TV, 0 for TVEQ, >0 for TVDN with epsilon as noise level. Solving TVDN model takes much more time than unconstraint TV. [-1, unconstraint TV]", false, -1, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "factor to the penalty coefficient for data fidelity. The most important and sensible parameter. Increasing mu gives more importance to data fidelity term. Suggested values:\n\
pix=256: 5e-2 for data of SNR 50db, 5e-3 for data of SNR 25db\npix=512: 1e-2 for data of SNR 50 db, 1e-3 for data of SNR 25db. [5e-2]", false, 5e-2, "double", cmd);
  TCLAP::ValueArg<double> betaArg("", "beta", "penalty coefficient for gradient penalty. Larger beta accelerate the convergence but reduce the precision. [100]", false, 100, "double", cmd);
  TCLAP::ValueArg<double> tolInnArg("", "tolInn", "inner iterations stopping tolerance. Decrease for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "outer(global) iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> rwEpsilonArg("", "rwEpsilon", "reweighting parameter between (0, 1], meaningful only for rwIter>1. A grid node is treated as contour if its gradient norm is proportionally bigger than epsilon. Use small epsilon for strong edge detection/preservation behavior. [1]", false, 1, "double", cmd);

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

  AlgoParms AP;			// Algo structure
  AP.pix = pixArg.getValue();

  AP.nonneg = nonnegSwitch.getValue();
  AP.maxIter = IterArg.getValue();
  AP.tolInn = tolInnArg.getValue();
  AP.tol = tolArg.getValue();
  AP.mu = muArg.getValue();
  AP.beta = betaArg.getValue();
  AP.aniso = anisoSwitch.getValue();
  AP.epsilon = epsilonArg.getValue();
  //AP.eq = eqSwitch.getValue();

  AP.rwIter = rwIterArg.getValue();
  AP.rwEpsilon = rwEpsilonArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Create output path
  char buffer[256];
  sprintf(buffer, "PixTV.at_pix[%d]_mu[%1.1e]_tol[%1.1e]_epsilon[%1.1e]_rwIter[%d].%s", AP.pix, AP.mu, AP.tol, AP.epsilon, AP.rwIter, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname);

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);  
  // Normalize the sinogram 
  double Ynrml = Y.abs().mean(); 
  Y /= Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its maximum coefficient : "<<Ynrml<<endl;

  AP.pixSize = conf.sizeObj.maxCoeff() / AP.pix;
  Array2d sz = conf.sizeObj + AP.pixSize/2;
  AP.dimObj = Array2i((int)(sz.x() / conf.sizeObj.maxCoeff() * AP.pix), (int)(sz.y() / conf.sizeObj.maxCoeff() * AP.pix));

  // Projector
  PixDrvProjector *P = new PixDrvProjector(conf, AP.dimObj, AP.pixSize);
  // Gradient operator
  PixGrad *G = new PixGrad(AP.dimObj, 1);

  double norm_projector = P->estimate_opNorm();
  DiagOp *M = new DiagOp(P->get_dimY(), 1./norm_projector);
  CompOp *A = new CompOp(M, P);
  if (verbose)
    cout<<"Projector is normalized by the estimated norm: "<<norm_projector<<endl;

 // Normalization of mu
  AP.mu_rel = AP.mu;
  AP.mu *= AP.dimObj.prod();

  // Initialization
  //ArrayXd Xr = P->backward(Y);
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

      if (verbose > 1) {	      
	imr = Map<ImageXXd>(Xr.data(), AP.dimObj.y(), AP.dimObj.x());
      	sprintf(buffer, "%s/rwIter_%d", outpath_fname.c_str(), n);
      	CmdTools::imsave(imr, buffer);

      	sprintf(buffer, "%s/edge_rwIter_%d", outpath_fname.c_str(), n);
	ImageXXd wg = Map<ImageXXd>(W.data(), AP.dimObj.y(), AP.dimObj.x());
      	CmdTools::imsave(wg, buffer);
      }
    }
  }

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization and descaling Xr
  Xr *= (Ynrml / norm_projector);

  // SNR and Corr of reconstruction
  if (conf.phantom != "") {
    ImageXXd im = CmdTools::imread(conf.phantom.data());
    imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), im.rows(), im.cols());
    // double pixSize = fmax(AP.dimObj.x() *1./ im.cols(), AP.dimObj.y() *1./ im.rows());
    // imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), 1., im.rows(), im.cols(), pixSize);
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }
  else {
    if (conf.dimObj.x() > 0) {    
      imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), conf.dimObj.y(), conf.dimObj.x());
      // double pixSize1 = fmax(conf.sizeObj.x() / conf.dimObj.x(), conf.sizeObj.y() /conf.dimObj.y());
      // imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), AP.pixSize, conf.dimObj.y(), conf.dimObj.x(), pixSize1);
    }
    else
      imr = Map<ImageXXd>(Xr.data(), AP.dimObj.y(), AP.dimObj.x());
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(Xr, AP.dimObj.y(), AP.dimObj.x(), buffer);
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer); // Save also in binary

  // Save reconstructed image
  sprintf(buffer, "%s/recon_resampled", outpath_fname.c_str());
  CmdTools::imsave(imr, buffer);

  // Save reconstruction parameters
  save_algo_parms(outpath_fname+"/parameters.cfg", AP, msg);
  if (bcsv_fname != "") {
    save_algo_parms_batch_csv(bcsv_fname, AP);
  }

  cout<<"Outputs saved in directory "<<outpath_fname<<endl;
  
  if (verbose > 1) {
    CmdTools::imshow(imr, "Reconstructed pixel image");
  }

  return 0;
}

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg *msg)
{
  // Save current algorithm's parameters in a file
  ofstream fout; //output configuration file
  fout.open(fname.data(), ios::out);
  if (!fout) {
    cout <<"Cannot open file : "<<fout<<endl;
    exit(1);
  }

  fout<<"[Pixel image Parameters]"<<endl;
  fout<<"rows="<<AP.dimObj.y()<<endl;
  fout<<"cols="<<AP.dimObj.x()<<endl;
  fout<<"rows*cols="<<AP.dimObj.x()*AP.dimObj.y()<<endl;
  fout<<"pixSize="<<AP.pixSize<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
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
    fout<<"residual="<<msg[n].res<<endl;
    fout<<"tvnorm="<<msg[n].norm<<endl;
  }

  fout.close();
}

void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP)
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
    fp<<"PixTV_batch_reconstruction_results"<<endl;
    // Acquisition
    fp<<"nbProj, "<<"pixDet, ";
    // BlobImage
    fp<<"nbPix, "<<"rows, "<<"cols, "<<"pixSize, ";
    // Algo related
    fp<<"mu_rel, snr, corr, uqi, mse, si, time"<<endl;
  }

  fp<<AP.nbProj<<", "<<AP.pixDet<<", ";
  fp<<AP.dimObj.prod()<<", "<<AP.dimObj.y()<<", "<<AP.dimObj.x()<<", "<<AP.pixSize<<", ";
  fp<<AP.mu_rel<<", "<<AP.snr<<", "<<AP.corr<<", "<<AP.uqi<<", "<<AP.mse<<", "<<AP.si<<", "<<AP.time;
  fp<<endl;

  fp.close();  
}


  //  {
  //   ArrayXd X0, Y0, Z0;
  //   X0 = ArrayXd::Random(P->get_dimX());
  //   Z0 = ArrayXd::Random(P->get_dimY());
  //   cout<<"<Z, P(X)> = "<<(Z0 * P->forward(X0)).sum()<<endl;
  //   cout<<"<Pt(Z), X> = "<<(P->backward(Z0) * X0).sum()<<endl;

  //   X0 = ArrayXd::Random(G->get_dimX());
  //   Z0 = ArrayXd::Random(G->get_dimY());
  //   cout<<"<Z, G(X)> = "<<(Z0 * G->forward(X0)).sum()<<endl;
  //   cout<<"<Gt(Z), X> = "<<(G->backward(Z0) * X0).sum()<<endl;
  // }

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

  //     // X0 = ArrayXd::Random(G->get_dimX());
  //     // Z0 = ArrayXd::Random(G->get_dimY());
  //     // err = (Z0 * G->forward(X0)).sum()-(G->backward(Z0) * X0).sum();
  //     // printf("<Z, G(X)> - <Gt(Z), X> = %1.10e\n\n", err);
  //   }
  // }


  // string im_fname = im_fnameArg.getValue(); // Test grad operator
  // if (im_fname != "") {
  //   ImageXXd im = CmdTools::imread(im_fname.data());
  //   Array2i dimObj(im.cols(), im.rows());

  //   PixGrad *G = new PixGrad(dimObj);

  //   vector<ImageXXd> DX = G->imgrad(im);
  //   ImageXXd DtDX = G->imgradT(DX);
  //   CmdTools::multi_imshow(DX, "grad");
  //   CmdTools::imshow(DtDX, "gradT");
  // }


