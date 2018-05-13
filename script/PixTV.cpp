// Image reconstruction by TV minimization method and pixel representation 
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
      min_x TV(x), st. |Ax-b|^2<M*epsilon^2, and x>=0  (TVDN)\n\
Examples:\n\
1. PixTV.run DIR --pix 256 --tol 1e-3 --iter 500 -vv\n\
    solves the problem (TV), on a image of dimension 256^2, with tolerance 1e-3 and 500 iterations.\n\
2. PixTV.run DIR --pix 512 --epsilon 1e-2 --rwIter 4 --rwEpsilon 0.1 -vv\n\
    solves the problem (TVDN, epsilon=1e-2) using 4 reweighted multistep iterations, on a image of dimension 512^2 with tolerance 1e-4, the intermediate results are saved.");
   
  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);

  TCLAP::ValueArg<int> BenchArg("", "bench", "Bench mark test (only) for operator. [0: no bench mark test]", false, 0, "int", cmd);
  TCLAP::ValueArg<int> pixArg("", "pix", "reconstruction dimension of pixel image (larger side). [512]", false, 512, "int", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. For box=1, the square ROI is the smallest square including circle ROI. [0.7]", false, 0.7, "double", cmd);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::SwitchArg anisoSwitch("", "aniso", "anisotropic TV model. For piecewise constant image (like a square) anisotropic TV can give much better result than TV. On noisy data or natural image this has no visible effect. [false]", cmd, false);
  TCLAP::SwitchArg nonnegSwitch("", "nonneg", "turn-on positive constraint. Positivity constraint may accelerate the convergence. [false]", cmd, false);

  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "model selection: <0 for unconstraint TV, 0 for TVEQ, >0 for TVDN with epsilon as noise level per detector, i.e. |Ax-b|^2<M*epsilon^2 with M the sinogram dimension.\
 Solving TVDN model takes much more time than unconstraint TV. [-1, unconstraint TV; 5e-3 for TVDN of data SNR 50db]", false, -1, "double", cmd);

//   TCLAP::ValueArg<double> muArg("", "mu", "the penalty coefficient for data fidelity. The most important and sensible parameter. Increasing mu gives more importance to data fidelity term. Some suggestions:\n\
// It seems that the optimal choice of mu depends greatly on the acquisition configuration (pixDet, nbProj). The default value mu=100 works well on high SNR (>=50db) simulation data of nbProj>=64, pixDet>=512,\
//  while for nbProj small (e.g. 8 projections only) mu=100 can totally fail, and mu=5 in this case gives much good result. It depends also on the the dimesion of unknown (larger mu for more unknowns) but in a less sensitive manner. [100]", false, 100, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "The most important parameter, it has two meanings:\n\
Unconstraint TV model(epsilon<0): this is the penalization of data fitness(its value will be automatically scaled). It depends inverse proportionally on the noise level, \
and on the complexity of image. On 50db SNR data, in function of image type it can varies between [5e3, 5e6]. Empirically: ~1e4 for simple geometric object, ~1e6 for complexe natural image. \n\
Constraint TV model(epsilon>=0): this is the Augmented Lagrangian penalty coefficient, and it should take the same order of value as beta. [1e6]", false, 1e6, "double", cmd);

  TCLAP::ValueArg<double> betaArg("", "beta", "penalty coefficient for gradient penalty. Larger beta accelerate the convergence but reduce the precision. [100]", false, 100, "double", cmd);
  TCLAP::ValueArg<double> tolInnArg("", "tolInn", "inner iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> tolGapArg("", "tolGap", "TVAL3 : stopping critera in terms of duality gap. [2e-2]", false, 2e-2, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "outer(global) iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> rwEpsilonArg("", "rwEpsilon", "reweighting parameter between (0, 1], meaningful only for rwIter>1. A grid node is treated as contour if its gradient norm is proportionally bigger than rwEpsilon. Use small value for strong edge detection/preservation behavior. [1]", false, 1, "double", cmd);

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
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();

  AP.nonneg = nonnegSwitch.getValue();
  AP.maxIter = IterArg.getValue();
  AP.tolInn = tolInnArg.getValue();
  AP.tolGap = tolGapArg.getValue();
  AP.tol = tolArg.getValue();
  AP.mu = muArg.getValue();
  AP.beta = betaArg.getValue();
  AP.aniso = anisoSwitch.getValue();
  AP.epsilon = epsilonArg.getValue();

  AP.rwIter = rwIterArg.getValue();
  AP.rwEpsilon = rwEpsilonArg.getValue();

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

  // Set GPU device
  Tools::setActiveGPU(gpuid);

  SimConfig conf = CmdTools::load_acqcarm(cfg_fname, AP.box, AP.roi);

  // Load data from binary file
  ArrayXd Y = CmdTools::loadarray(sg_fname, conf.nbProj_total * conf.pixDet);  
  AP.Ynrml = (AP.sgNrml) ? Y.abs().mean() : 1.; // normalization factor to compensate the sinogram value impact
  Y /= AP.Ynrml;
  if (verbose)
    cout<<"Sinogram is normalized by its mean abs value: "<<AP.Ynrml<<endl;

  AP.pixSize = conf.sizeObj.maxCoeff() / AP.pix;
  AP.nbProj = conf.nbProj_total;
  Array2d sz = conf.sizeObj + AP.pixSize/2;
  AP.dimObj = Array2i((int)(sz.x() / conf.sizeObj.maxCoeff() * AP.pix), (int)(sz.y() / conf.sizeObj.maxCoeff() * AP.pix));

  // Projector
  PixDrvProjector *P = new PixDrvProjector(conf, AP.dimObj, AP.pixSize);
  // Gradient operator
  PixGrad *G = new PixGrad(AP.dimObj, 1);

  //if (AP.epsilon<0) 
  {
    // Explenation: The true TV norm of pixel image(as a piecewise
    // constant function) ~= h *(discrete TV norm), with h the pixel
    // side size (see ch3, sec1.3 of the thesis). Therefore optimal mu
    // should behave proportionally to 1/h. At the same time the data
    // fitness term |Af-b|^2 ~= M*epsilon, with M the dimension of
    // data and epsilon the noise level of each detector
    // bin. Therefore optimal mu should behave proportionally to 1/M.

    double mu_factor = 1 / AP.pixSize / (1. * conf.nbProj_total * conf.pixDet);
    cout<<"Scaling mu by the factor: "<<mu_factor<<endl;
    AP.mu_rel = AP.mu;
    AP.mu *= mu_factor;
  }
  // else {
  //   cout<<"Constraint TV minimization model: mu should be set to the same order as beta!"<<endl;
  //   AP.mu = AP.beta;
  // }

  if (BenchArg.getValue()>0) {	// Bench-mark test
    int N = BenchArg.getValue();
    cout<<"Pix-driven projector Bench-mark test:"<<endl;
    CmdTools::BenchMark_Op(*P, N);
    cout<<"Pixel gradient Bench-mark test:"<<endl;
    CmdTools::BenchMark_Op(*G, N);
    exit(0);
  }

  // Create output path
  char buffer[256];
  sprintf(buffer, "PixTV_pix[%d]_mu[%1.1e]_tol[%1.1e]_epsilon[%1.1e]_rwIter[%d].%s", AP.pix, AP.mu_rel, AP.tol, AP.epsilon, AP.rwIter, pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

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
    //    ArrayXd Nu, Lambda;

    for(int n=0; n<AP.rwIter; n++) {
      if (verbose)
	cout<<"\nReweighted TV minimization iteration : "<<n<<endl;    

      // Lambda and Nu should NOT be reused.
      // Lambda.setZero(P->get_dimY());
      // Nu.setZero(G->get_dimY());

      msg[n] = SpAlgo::TVAL3(*P, *G, W, Y, Xr, //Nu, Lambda, 
      			     (AP.epsilon<0) ? -1 : AP.epsilon*sqrt(conf.nbProj_total*conf.pixDet), // Noise level epsilon is scaled
			     AP.aniso, AP.nonneg, AP.mu, AP.beta, 
      			     AP.tolInn, AP.tol, AP.tolGap, AP.maxIter, verbose);

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
  Xr *= AP.Ynrml;

  // SNR and Corr of reconstruction
  if (conf.phantom != "") {
    ImageXXd im = CmdTools::imread(conf.phantom.data());
    imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), im.rows(), im.cols(), false);
    // double pixSize = fmax(AP.dimObj.x() *1./ im.cols(), AP.dimObj.y() *1./ im.rows());
    // imr = Tools::resampling_piximg(Xr.data(), AP.dimObj.y(), AP.dimObj.x(), 1., im.rows(), im.cols(), pixSize);
    AP.snr = Tools::SNR(imr, im);
    AP.corr = Tools::Corr(imr, im);
    AP.uqi = Tools::UQI(imr, im);
    AP.mse = (imr-im).matrix().squaredNorm() / im.size();
    AP.si = Tools::StreakIndex(imr, im);
    printf("UQI = %f\tMSE = %f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }
  else {
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

  // Change the output directory name if SNR is available
  if (conf.phantom != "") {
    sprintf(buffer, "%s.snr[%2.2f]", outpath_fname.c_str(), AP.snr);
    outpath_fname = CmdTools::rename_outpath(outpath_fname, buffer);
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
  fout<<"size of pixel="<<AP.pixSize<<endl;
  fout<<"box="<<AP.box<<endl;
  fout<<"Ynrml="<<AP.Ynrml<<endl;
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
    fout<<"normalized residual |Ax-b|="<<msg[n].res<<endl;
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


