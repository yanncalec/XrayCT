// Image reconstruction by Wavelet l1 prior minimization method and pixel representation (with tunning of projector)
// The PADM algorithm is believed to converge faster than FISTA. Nevertheless, this algorithm is much complicate to implement,
// and it's sensitive to the choice of parameters. The projector should be normalized to guarantee the convergence.
// For PADM, mu and beta together rule the reconstruction. Smaller beta, stronger shrinkage. Smaller mu, stronger data fidelity.
// The empirical values mu, beta have been tested for nbProj=128, pixDet=512 and 50db sinogram. Unlike Fista, these values(especially, mu)
// have different interpretation and should not be scaled.
// The multi-step reweighting iteration is more stable when it's applied on the constraint problem, and with the reweighting coeffcient value normalized to 1.

#include "CmdTools.hpp"
#include "T2D.hpp"
#include <tclap/CmdLine.h>

void save_algo_parms(const string &fname, const AlgoParms &AP, const ConvMsg *msg);
void save_algo_parms_batch_csv(const string &fname, const AlgoParms &AP);

int main(int argc, char **argv) {

   TCLAP::CmdLine cmd("2D Reconstruction by L1 minimization on Wavelet representation. \
Solve:\n\
     min |AD*x-b|^2 + mu*|x|_1 (l1)\n\
     min |x|_1 st ADx=b (BP)\n\
     min |x|_1 st |AD*x-b| < epsilon (BPDN)\n\
A is the X-ray projector, D is the wavelet synthesis operator, and x is the wavelet coefficient.\
The l1 norm can be reweighted, which is useful in the multistep l1 reconstruction.\n\
Examples:\n\
     PixL1.run DIR --pix 512 --wvl db --order 6 --algo padm --beta 200 --tol 1e-3 --iter 500 -v\n\
solves the problem (l1) using db6 wavelet on a Cartesian grid of dimension 512X512, tolerance 1e-3 and 500 iterations. The algorithm in use is padm.\n\
     PixL1.run DIR --pix 512 --rwIter 3 --epsilon 0 --rwEpsilon 0.1 --tol 1e-3 --iter 500 -vv\n\
solves the problem (BP) by a 3 steps reweighted l1 iteration, and save the intermediate results.\n\
     PixL1.run DIR --xr xr.dat --pix 512 --epsilon 0 --rwIter 3 --rwEpsilon 0.1 --tol 1e-3 --iter 500\n\
solves the problem (BP) by a 3 steps reweighted l1 iteration, initialized by a reconstructed image (of dimension 512^2) stored in binary file xr.dat.\n");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);
  TCLAP::ValueArg<string> pfixArg("", "pfix", "Post-fix patched to output path. [SYSNAME]", false, "", "string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);
  TCLAP::ValueArg<string> xr_fnameArg("", "xr","reconstructed file name. Reuse the reconstructed coeffcient for reweighted l1 norm",false,"","string", cmd);

  TCLAP::ValueArg<int> pixArg("", "pix", "reconstruction dimension of pixel image (larger side). [512]", false, 512, "int", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. For box=1, the square ROI is the smallest square including circle ROI. [0.7]", false, 0.7, "double", cmd);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<string> wvlArg("", "wvl", "wavelet name: haar, db. [db]", false, "db", "string", cmd);
  TCLAP::ValueArg<int> orderArg("", "order", "order of daubechies wavelet: 2, 4...20. [6]", false, 6, "int", cmd);
  //TCLAP::ValueArg<double> nzArg("", "nz", "The percentage (between 0 and 1) of the biggest coeffcients. This defines a support set S and the algorithm exits when S is stablized. nz=0 or 1 turns off the constraint.  [0]", false, 0, "double", cmd);
  //TCLAP::SwitchArg debiasSwitch("", "debias", "MSE debiasing on reconstruction using support information [false]", cmd, false);

  TCLAP::ValueArg<double> betaArg("", "beta", "PADM : normalized data fitnesss penalty, small beta for large shrinkage effect. [7.5e3]", false, 7.5e3, "double", cmd);
  //  TCLAP::ValueArg<double> betaArg("", "beta", "PADM : normalized data fitnesss penalty, small beta for large shrinkage effect. [200]", false, 200, "double", cmd);
  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "PADM : model selection: 0 for BPEQ, >0 for BPDN with epsilon as noise level per detector, i.e. |Ax-b|^2<M*epsilon^2 with M the sinogram dimension. [5e-3 for data of SNR 50db]", false, 5e-3, "double", cmd);
  //  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "PADM : model selection: <0 for unconstraint l1, 0 for BPEQ, >0 for BPDN with epsilon as noise level per detector, i.e. |Ax-b|^2<M*epsilon^2 with M the sinogram dimension.", false, -1, "double", cmd);
  TCLAP::ValueArg<double> tauArg("", "tau", "PADM : proximal penalty. [1.]", false, 1., "double", cmd);
  TCLAP::ValueArg<double> gammaArg("", "gamma", "PADM : lagrangian step. [1.]", false, 1., "double", cmd);

  TCLAP::ValueArg<double> tolArg("", "tol", "iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> maxIterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> rwEpsilonArg("", "rwEpsilon", "reweighting parameter between (0, 1], meaningful for rwIter>1. Wavelet coefficients bigger than epsilon are treated as support, so small epsilon for strong support detection behavior. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<int> gpuidArg("g", "gpuid", "ID of gpu device to use. [0]", false, 0, "int", cmd);
  TCLAP::MultiSwitchArg verboseSwitch("v", "verbose", "Print informations and display image. [quite]", cmd);
  TCLAP::ValueArg<int> FreqArg("", "mfreq", "print convergence message every mfreq iterations. [50]", false, 50, "int", cmd);

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
  AP.pix = pixArg.getValue();
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();

  AP.wvlname = wvlArg.getValue();
  AP.wvlorder = (AP.wvlname == "haar") ? 2 : orderArg.getValue();
  // AP.nz = nzArg.getValue();
  // AP.debias = debiasSwitch.getValue();
  
  AP.beta = betaArg.getValue();
  AP.epsilon = epsilonArg.getValue();
  AP.tau = tauArg.getValue();
  AP.gamma = gammaArg.getValue();
  AP.maxIter = maxIterArg.getValue();
  AP.tol = tolArg.getValue();
  AP.algo = "padm";

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

  AP.pixSize = conf.sizeObj.maxCoeff() / AP.pix;
  Array2d sz = conf.sizeObj + AP.pixSize/2;
  AP.dimObj = Array2i((int)(sz.x() / conf.sizeObj.maxCoeff() * AP.pix), (int)(sz.y() / conf.sizeObj.maxCoeff() * AP.pix));

  // Projector and Wavelet operator
  PixDrvProjector *P = new PixDrvProjector(conf, AP.dimObj, AP.pixSize);
  GSL_Wavelet2D *D = new GSL_Wavelet2D(AP.dimObj, AP.wvlname, AP.wvlorder);

  // For padm algorithm, the projector must be normalized to guarantee the convergence?
  // Fista dosen't need the normalization, but the mu can be proportionally scaled to work with the normalized projector.
  //LinOp *A = new CompOp(P, D), *M;
  //AP.norm_projector = 1.;
  // Normalize the projector
  if (verbose)
    cout<<"Estimation of projector norm..."<<endl;
  AP.norm_projector = P->estimate_opNorm();
  if (verbose)
    cout<<"Projector is normalized by the estimated norm: "<<AP.norm_projector<<endl;
    
  LinOp *M = new DiagOp(P->get_dimY(), 1./AP.norm_projector); // diagonal operator which normalize the projector 
  LinOp *A = new CompOp(new CompOp(M, P), D);		      // composition operator PD

  // Normalization of mu
  AP.beta_rel = AP.beta;
  double beta_factor = AP.norm_projector / (conf.pixDet * conf.nbProj_total * log(P->get_dimX())); 
  AP.beta *= beta_factor;
  cout<<"Scaling beta by the factor: "<<beta_factor<<endl;

  // Create output path
  char buffer[256];
  sprintf(buffer, "PixL1PADM_pix[%d]_wvl[%s]_order[%d]_beta[%1.1e]_epsilon[%1.1e]_tol[%1.1e]_rwIter[%d]_rwEpsilon[%2.2f]_algo[%s].%s", AP.pix, AP.wvlname.c_str(), AP.wvlorder, AP.beta_rel, AP.epsilon, AP.tol, AP.rwIter, AP.rwEpsilon, AP.algo.c_str(), pfix.c_str());
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // Initialization
  ArrayXd Xr, Cr, W;
  ArrayXd toto;
  ImageXXd imr;			// Sum of reconstructed image

  // If a data file is passed, use it as initialization and as reweighting coefficient.
  if (xr_fname != "") {
    cout<<"Load reconstructed image and calculate the wavelet coefficient..."<<endl;
    Xr = CmdTools::loadarray(xr_fname, AP.dimObj.x()*AP.dimObj.y());
    Cr = D->backward(Xr);
    toto = Cr.abs();
    double gmax = toto.maxCoeff();
    for (size_t m=0; m<toto.size(); m++)
      toto[m] = 1/(toto[m] + AP.rwEpsilon * gmax);
    W = toto / toto.maxCoeff();
    //printf("W.min=%f, W.max=%f\n", W.minCoeff(), W.maxCoeff());    
  }
  else {
    Cr = ArrayXd::Zero(D->get_dimX());
    W = ArrayXd::Ones(Cr.size());			       // Reweighting coefficient
  }
  ConvMsg* msg = new ConvMsg[AP.rwIter];
  clock_t t0 = clock();
  
  ArrayXd Z;			// Lagrangian multiplier
  ArrayXd R;			// Residual or the auxilary variable
  Z.setZero(Y.size()); 
  R.setZero(Y.size()); 

  // The following is the reweighted iteration  
  for(int n=0; n<AP.rwIter; n++) {
    if (verbose)
      cout<<"\nReweighted L1 minimization iteration : "<<n<<endl;    
    
    if (AP.algo == "padm")
      msg[n] = SpAlgo::L1PADM(*A, W, Y, Cr, Z, R,
			      AP.mu, AP.epsilon, AP.beta, AP.tau, AP.gamma, 
			      AP.tol, AP.maxIter, 0, 0, verbose, Freq);

    // update reweighting coefficeint
    toto = Cr.abs();
    double gmax = toto.maxCoeff();
    for (size_t m=0; m<toto.size(); m++)
      toto[m] = 1/(toto[m] + AP.rwEpsilon * gmax);
    W = toto / toto.maxCoeff();
    //printf("W.min=%f, W.max=%f\n", W.minCoeff(), W.maxCoeff());    

    // ArrayXd Mask = (W <= AP.rwEpsilon * gmax / pow(2., n+1.)).select(1, ArrayXd::Zero(W.size())); // support mask
    // ArrayXd toto(W.size());
    // for (size_t m=0; m<W.size(); m++)
    //   toto[m] = 1/(W[m] + AP.rwEpsilon * gmax);
    // W = toto * Mask / toto.maxCoeff(); // This works the best

    if (verbose > 1) {	      
      Xr = D->forward(Cr);
      imr = Map<ImageXXd>(Xr.data(), AP.dimObj.y(), AP.dimObj.x());
      sprintf(buffer, "%s/rwIter_%d", outpath_fname.c_str(), n);
      CmdTools::imsave(imr, buffer);
    }
  }

  AP.time = (clock()-t0)/(double)CLOCKS_PER_SEC;
  printf("Reconstruction taken %lf seconds\n", AP.time); 

  // Denormalization
  Cr *= (AP.Ynrml / AP.norm_projector);
  //Cr /= AP.norm_projector;
  AP.nnz = Tools::l0norm(Cr);
  AP.sparsity = AP.nnz*1./ Cr.size(); // Be careful, this value may be too small due to the zero padding.
  Xr = D->forward(Cr);

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
    printf("Sparsity = %f\tUQI = %f\tMSE=%f\tCorr. = %f\tSNR = %f\tSI = %f\n", AP.sparsity, AP.uqi, AP.mse, AP.corr, AP.snr, AP.si);
  }
  else {
    imr = Map<ImageXXd>(Xr.data(), AP.dimObj.y(), AP.dimObj.x());
  }

  // Save reconstructed coefficients Xr
  sprintf(buffer, "%s/recon", outpath_fname.c_str());
  CmdTools::imsave(Xr, AP.dimObj.y(), AP.dimObj.x(), buffer);
  sprintf(buffer, "%s/xr", outpath_fname.c_str());
  CmdTools::savearray(Xr, buffer); // Save also in binary
  sprintf(buffer, "%s/cr", outpath_fname.c_str());
  CmdTools::savearray(Cr, buffer); // Save wavelet coefficients in binary

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
  fout<<"box="<<AP.box<<endl;
  fout<<"Ynrml="<<AP.Ynrml<<endl;
  fout<<"norm_projector="<<AP.norm_projector<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"wvlname="<<AP.wvlname<<endl;
  fout<<"wvlorder="<<AP.wvlorder<<endl;
  fout<<"besov="<<AP.besov<<endl;
  if(AP.algo=="padm") {
    fout<<"beta_rel="<<AP.beta_rel<<endl;
    fout<<"beta="<<AP.beta<<endl;
    fout<<"epsilon="<<AP.epsilon<<endl;
    fout<<"gamma="<<AP.gamma<<endl;
    fout<<"tau="<<AP.tau<<endl;
  }
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
    fout<<"normalized residual |Ax-b|="<<msg[n].res<<endl;
    fout<<"l1 norm="<<msg[n].norm<<endl;
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
    fp<<"PixL1_batch_reconstruction_results"<<endl;
    // Acquisition
    fp<<"nbProj, "<<"pixDet, ";
    // BlobImage
    fp<<"nbPix, "<<"rows, "<<"cols, "<<"pixSize, ";
    // Algo related
    fp<<"beta_rel, beta, snr, corr, uqi, mse, si, time"<<endl;
  }

  fp<<AP.nbProj<<", "<<AP.pixDet<<", ";
  fp<<AP.dimObj.prod()<<", "<<AP.dimObj.y()<<", "<<AP.dimObj.x()<<", "<<AP.pixSize<<", ";
  fp<<AP.beta_rel<<", "<<AP.beta<<", "<<AP.snr<<", "<<AP.corr<<", "<<AP.uqi<<", "<<AP.mse<<", "<<AP.si<<", "<<AP.time;
  fp<<endl;

  fp.close();  
}

  //AP.mu /= A->get_dimX();
  //AP.beta *= 2 * Y.size() / Y.abs().sum();
  //AP.beta *= sqrt(A->get_dimX());
  //AP.beta = P->get_dimY()*2./Y.abs().sum(); // This is the value used in the original paper


  // AP.mu_rel = AP.mu;
  // AP.beta_rel = AP.beta;
  // if (AP.epsilon<0) {		// Meaningful only when unconstraint BP model is used
  //   // D->get_nbScale() is due to the implementation of GSL
  //   //double mu_factor = (1. * conf.pixDet * conf.nbProj_total) / (D->get_nbScale() - 1.) / AP.norm_projector; 
  //   double mu_factor = (conf.pixDet * conf.nbProj_total) * log(D->get_dimX()) / AP.norm_projector; 
  //   cout<<"Scaling mu by the factor: "<<mu_factor<<endl;
  //   AP.mu *= mu_factor;
  // }
  // else {
  //   cout<<"Constraint BP model: mu has no effect."<<endl;
  // }
