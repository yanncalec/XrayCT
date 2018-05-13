// Image reconstruction by TV minimization method and blob representation

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
     min_f TV(f), st. |Af-b|^2<M*epsilon^2, and f>=0  (TVDN), M is the dimension of sinogram vector.\n\
Here TV(f) is the discrete TV norm of a blob image represented by blob coefficients f_1..f_N.\n\
About blob image:\n\
The blob image f(x) is a function of type:\n\
- Single scale: f(x)=sum_k f_k phi(x-x_k)\n\
- Multiscale: f(x) = sum_j sum_k f_{jk} phi_j(x-x_{jk})\n\
{f_k}, {f_{jk}} are blob coefficients, {phi(x-x_k)}, {phi(x-x_{jk})} are blobs centered at node x_k and x_{jk} (can be hexagonal or Cartesian lattice).\n\
Parameters of a blob image:\n\
Due to the localization character of blob, the reconstruction with a blob image is more stable (less high frequency error, faster convergence) than a pixel image of equivalent spatial resolution. \
Some important parameters of a blob image include:\n\
1. Number of scales. A multiscale blob image is composed of a coarse scale Gaussian blob image and several fine scale blob images of type X. The total number of scales is passed by --nbScale N (N>=1).\
TV minimization on a multiscale model has no advantage than a single scale model. It's implemented only for educational purpose.\n\
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
    1. BlobTV.run DIR --fspls 4 --nbScale 1 --mu 1e6 --tol 1e-3 --iter 500 -vv\n\
solves the problem (TV) using a single scale Gaussian blob image on a hexagonal grid, with tolerance 1e-3 and 500 iterations. mu=1e6 is a factor of penalization(true value of mu is auto-scaled), the bandwidth of reconstruction is 1/4 of the optimal one.\n\
    2. BlobTV.run DIR --cat --fspls 2 --nbScale 1 --mu 1e6 --epsilon 0.1 --tol 1e-3 --iter 500 -vv\n\
solves the problem (TVDN) using a single scale Gaussian blob image on a Cartesian grid, the bandwidth of reconstruction is 1/2 of the optimal one.\n\
    3. BlobTV.run DIR --fspls 4 --nbScale 1 --mu 1e6 --rwIter 3 --rwEpsilon 1 --tol 1e-3 --iter 500 -vv\n\
solves the problem (TV) with 3 reweighted iterations.\n\
\nSuggestions on the choice of regularization parameter mu:\n\
Infortunately, there's no very scientific rules in setting mu, the most important parameter. We suggest the following manual method for the unconstraint TV model. Fix the acquisition configuration, the noise level and the blob image parameters, in command line run:\n\
        for MU in a0 a1...aN; do BlobTV.run --fspls X --mu $MU --bcsv test.csv ; done\n\
This will test a range of mu (e.g. in [5e3, 5e6]) and write the reconstruction results (SNR etc) in a csv file test.csv. Choose manually the best mu then and use it for the current configuration.");

  TCLAP::UnlabeledValueArg<string> inpath_fnameArg("inpath_fname","Input data directory name",true,"","string", cmd);
  TCLAP::UnlabeledValueArg<string> outpath_fnameArg("outpath_fname","Relative output data directory name [auto.]",false,"","string", cmd);
  TCLAP::ValueArg<string> bcsv_fnameArg("", "bcsv","CSV file name for batch test",false,"","string", cmd);

  TCLAP::ValueArg<string> cfg_fnameArg("", "cfg","Acquisition configuration file name",false,"acq.cfg","string", cmd);
  TCLAP::ValueArg<string> sg_fnameArg("", "sg","Sinogram data file name",false,"sino.dat","string", cmd);
  TCLAP::ValueArg<string> xr_fnameArg("", "xr","reconstructed file name. Reuse the reconstructed coeffcient for reweighted l1 norm",false,"","string", cmd);

  TCLAP::ValueArg<int> BenchArg("", "bench", "Bench mark test (only) for operator. [0: no bench mark test]", false, 0, "int", cmd);
  TCLAP::SwitchArg catSwitch("", "cat", "use cartesian grid for blob image and reconstruction. The hexagonal grid requires less samples (86.6%) than cartesian grid for representing a bandlimited function. [hexagonal grid]", cmd, false);

  TCLAP::ValueArg<double> roiArg("", "roi", "ratio of effective reconstruction ROI's diameter to the FOV's diameter. [1]", false, 1, "double", cmd);
  TCLAP::ValueArg<double> boxArg("", "box", "factor to diameter of the square ROI. By default(roi=1, box=1), the square ROI is the smallest square including circle ROI. [1]", false, 1, "double", cmd);

  TCLAP::ValueArg<double> fsplsArg("", "fspls", "factor to reconstruction bandwidth. Bigger fspls, smaller reconstruction bandwidth : fspls=2 for a blob two times bigper and a grid 2 times coarser. \
Thumb rule : 4 for pixDet=1024, 1 for pixDet=256, etc. Use small value (<=0.5) for insufficient detector resolution and oversampled angular resolution. [2]", false, 2, "double", cmd);
  TCLAP::ValueArg<int> upSplArg("", "upSpl", "Up-sampling factor for gradient grid. In theory upSpl>1 increases the presicion of TV norm approximation (and the computation load), but in practice it has no visible effect for various grid density. [1]", false, 1, "int", cmd);
  TCLAP::SwitchArg stripintSwitch("", "strip", "use blob strip-integral projector. [false]", cmd, false);
  TCLAP::SwitchArg nosgNrmlSwitch("", "nosgNrml", "do not normalize the sinogram [false]", cmd, false);

  TCLAP::ValueArg<int> nbScaleArg("", "nbScale", "number of scales in multi-scale blob image model. nbScale=1 for gaussian blob image. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> dilArg("", "dil", "scale dilation factor (scaling) for mexhat or d4gauss blob, meaningful only when nbScale>1. [2]", false, 2, "double", cmd);
  TCLAP::ValueArg<string> blobnameArg("", "blob", "name of detail blob profile, meaningful only when nbScale>1. Possible choices : 'mexhat', 'd4gs'. [mexhat]", false, "mexhat", "string", cmd);
  TCLAP::ValueArg<double> cut_off_errArg("", "cut", "cut-off error in space domain. [1e-3]", false, 1e-3, "double", cmd);
  TCLAP::ValueArg<double> fcut_off_errArg("", "fcut", "cut-off error in frequency domain, this is a fine-tuning for grid, it increases the grid density without dilating the blob. Decrease this value when the grain or gibbs artifacts are observed. This can seriously increase the computation time. [5e-2]", false, 5e-2, "double", cmd);
  TCLAP::SwitchArg frameSwitch("", "frame", "use frame but not tight frame multiscale system [false]", cmd, false);

  //TCLAP::SwitchArg tv2Switch("", "tv2", "using second order TV. [false]", cmd, false);
  TCLAP::SwitchArg anisoSwitch("", "aniso", "anisotropic TV model. [false]", cmd, false);
  TCLAP::SwitchArg nonnegSwitch("", "nonneg", "turn-on positive constraint. Positivity constraint may accelerate the convergence. It's automatically off if multiscale model is used. [false]", cmd, false);

  TCLAP::ValueArg<double> epsilonArg("", "epsilon", "model selection: <0 for unconstraint TV, 0 for TVEQ, >0 for TVDN with epsilon as noise level per detector pixel: epsilon*sqrt(sinogram dimension)=|Ax-b|^2\
 Solving TVDN model takes much more time than unconstraint TV. [-1, unconstraint TV; otherwise use BlobCG.run to estimate this value then double it]", false, -1, "double", cmd);

//   TCLAP::ValueArg<double> muArg("", "mu", "the penalty coefficient for data fidelity. The most important and sensible parameter. Increasing mu gives more importance to data fidelity term. Some suggestions:\n\
// It seems that the optimal choice of mu depends greatly on the acquisition configuration (pixDet, nbProj). The default value mu=100 works well on high SNR (>=50db) simulation data of nbProj>=64, pixDet>=512,\
//  while for nbProj small (e.g. 8 projections only) mu=100 can totally fail, and mu=5 in this case gives much good result. It depends also on the the dimesion of unknown (larger mu for more unknowns) but in a less sensitive manner. [100]", false, 100, "double", cmd);
  TCLAP::ValueArg<double> muArg("", "mu", "The most important parameter, it has two meanings:\n\
Unconstraint TV model(epsilon<0): this is the penalization of data fitness(its value will be automatically scaled). It depends inverse proportionally on the noise level, \
and on the complexity of image. On 50db SNR data, in function of image type it can varies between [5e3, 5e6]. Empirically for normalized sinogram: ~1e4 for simple geometric object, ~1e6 for complexe natural image. \n\
Constraint TV model(epsilon>=0): this is the Augmented Lagrangian penalty coefficient, and it should take the same order of value as beta. [1e6]", false, 1e6, "double", cmd);

  TCLAP::ValueArg<double> betaArg("", "beta", "penalty coefficient for gradient penalty. Larger beta accelerate the convergence but reduce the precision. [100]", false, 100, "double", cmd);
  TCLAP::ValueArg<double> tolInnArg("", "tolInn", "TVAL3 : inner iterations stopping tolerance. Decrease for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<double> tolGapArg("", "tolGap", "TVAL3 : stopping critera in terms of duality gap. [2e-2]", false, 2e-2, "double", cmd);
  TCLAP::ValueArg<double> tolArg("", "tol", "outer iterations stopping tolerance. Decrease it for high precision convergence. [1e-4]", false, 1e-4, "double", cmd);
  TCLAP::ValueArg<int> IterArg("", "iter", "maximum number of iterations. [1000]", false, 1000, "int", cmd);

  TCLAP::ValueArg<int> rwIterArg("", "rwIter", "number of reweighted iterations. [1]", false, 1, "int", cmd);
  TCLAP::ValueArg<double> rwEpsilonArg("", "rwEpsilon", "reweighting parameter between (0, 1], meaningful for rwIter>1. A grid node is treated as contour if its gradient norm is bigger than rWEpsilon. Use small value for strong edge detection/formation behavior. [1]", false, 1, "double", cmd);

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
  string xr_fname = xr_fnameArg.getValue();

  AP.hex = !catSwitch.getValue();
  AP.gtype = (AP.hex)?_Hexagonal_ : _Cartesian_;

  AP.roi = roiArg.getValue();
  AP.box = boxArg.getValue();
  AP.sgNrml = !nosgNrmlSwitch.getValue();

  AP.fspls = fsplsArg.getValue();
  AP.upSpl = upSplArg.getValue();

  AP.nbScale = nbScaleArg.getValue(); 
  AP.tightframe = !frameSwitch.getValue();
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
  AP.tolGap = tolGapArg.getValue();
  AP.tol = tolArg.getValue();
  AP.mu = muArg.getValue();
  AP.beta = betaArg.getValue();
  AP.aniso = anisoSwitch.getValue();
  AP.epsilon = epsilonArg.getValue();

  AP.rwIter = rwIterArg.getValue();
  AP.rwEpsilon = rwEpsilonArg.getValue();
  // Check that epsilon>=0 if rwIter>1
  if (AP.rwIter > 1 and AP.epsilon < 0) {
    cerr<<"Error: Reweighted TV iteration can be used only for epsilon>=0!"<<endl;
    exit(0);
  }

  int verbose = verboseSwitch.getValue();
  int gpuid = gpuidArg.getValue();

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
  // cout<<conf.sizeObj<<endl;
  // cout<<conf.diamROI<<endl;

  BlobImage *BI;
  if (AP.blobname == "gauss") {
    BI = BlobImageTools::SingleGauss(conf.sizeObj, conf.diamROI, AP.gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
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
  
  //Grid *igrid = new Grid(BI->bgrid[AP.nbScale-1]->sizeObj, BI->bgrid[AP.nbScale-1]->vshape*AP.upSpl, BI->bgrid[AP.nbScale-1]->splStep/AP.upSpl, AP.gtype, BI->bgrid[AP.nbScale-1]->diamROI); // Interpolation grid
  Grid *igrid = new Grid(BI->bgrid[0]->sizeObj, BI->bgrid[AP.nbScale-1]->vshape*AP.upSpl, BI->bgrid[AP.nbScale-1]->splStep/AP.upSpl, AP.gtype, BI->bgrid[0]->diamROI); // Interpolation grid
  // BlobImage *BI_W = BlobImageTools::SingleGauss(igrid->sizeObj, igrid->diamROI, igrid->gtype, 1./Acq::nrml_spDet(conf)/AP.fspls, AP.cut_off_err, AP.fcut_off_err);
  // cout<<igrid->nbNode<<endl;
  // cout<<BI->get_nbNode()<<endl;
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
  // It seems that even for the constraint TV model, mu has impact on the solution.
  double mu_factor = igrid->nbNode / pow(1.*STD_DIAMFOV, 2.) / (1.* conf.nbProj_total * conf.pixDet);
  cout<<"Scaling mu by the factor: "<<mu_factor<<endl;
  AP.mu_rel = AP.mu;
  AP.mu *= mu_factor;

  // if (AP.epsilon<0) {
  //   double mu_factor = igrid->nbNode / pow(1.*STD_DIAMFOV, 2.) / (1.* conf.nbProj_total * conf.pixDet);
  //   cout<<"Scaling mu by the factor: "<<mu_factor<<endl;
  //   AP.mu_rel = AP.mu;
  //   AP.mu *= mu_factor;
  // }
  // else {
  //   cout<<"Constraint TV minimization model: mu is set to the same order as beta!"<<endl;
  //   AP.mu_rel = AP.mu;
  //   //AP.mu = AP.beta;
  // }

  if (BenchArg.getValue()>0) {	// Bench-mark test
    int N = BenchArg.getValue();
    cout<<"Blob-driven projector Bench-mark test:"<<endl;
    CmdTools::BenchMark_Op(*P, N);
    cout<<"Blob-driven gradient Bench-mark test:"<<endl;
    CmdTools::BenchMark_Op(*G, N);
    exit(0);
  }

  // Initialization
  ArrayXd Xr = ArrayXd::Zero(P->get_dimX());
  ArrayXd W;			       // Reweighting coefficient
  ArrayXd Mask;			       // Reweighting mask
  vector<ImageXXd> Imr;		// Reconstructed multiscale image
  ImageXXd imr;			// Sum of reconstructed image
  
  int gradRank = G->get_shapeY().x(); // dimension of gradient vector at each site

  // If a data file is passed, use it as initialization and as reweighting coefficient.
  if (xr_fname != "") {
    cout<<"Load reconstructed blob coefficient..."<<endl;
    Xr = CmdTools::loadarray(xr_fname, BI->get_nbNode());

    if (AP.aniso)
      W = G->forward(Xr).abs();
    else
      W = SpAlgo::GTVNorm(G->forward(Xr), gradRank);
    double gmax = W.maxCoeff();
      
    Mask = (W <= AP.rwEpsilon * gmax / 2).select(1, ArrayXd::Zero(W.size())); // Edge mask
    ArrayXd toto(W.size());

    for (size_t m=0; m<W.size(); m++)
      toto[m] = 1/(W[m] + AP.rwEpsilon * gmax);
    W = toto  * Mask / toto.maxCoeff(); // This works the best
    printf("W.min=%f, W.max=%f\n", W.minCoeff(), W.maxCoeff());    
  }
  else {			// No data file, set W to one at firt iteration
    if (AP.aniso)
      W.setOnes(G->get_dimY());
    else 
      W.setOnes(G->get_dimY() / gradRank);
  }

  // Create output path
  char buffer[256];
  if (AP.rwIter>1) {
      sprintf(buffer, "BlobTV_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_epsilon[%1.1e]_pos[%d]_tol[%1.1e]_rwIter[%d]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err,  AP.nbScale, AP.dil, AP.blobname.c_str(), AP.epsilon, AP.nonneg, AP.tol, AP.rwIter, AP.roi, AP.box, pfix.c_str());
  }
  else{
    if (AP.epsilon >= 0)
      sprintf(buffer, "BlobTV_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_epsilon[%1.1e]_pos[%d]_tol[%1.1e]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err,  AP.nbScale, AP.dil, AP.blobname.c_str(), AP.epsilon, AP.nonneg, AP.tol, AP.roi, AP.box, pfix.c_str());
    else
      sprintf(buffer, "BlobTV_fspls[%2.2f]_fcut[%1.1e]_nbScale[%d]_dil[%2.1f]_blob[%s]_mu[%1.1e]_pos[%d]_tol[%1.1e]_roi[%1.2f]_box[%1.2f].%s", AP.fspls, AP.fcut_off_err, AP.nbScale, AP.dil, AP.blobname.c_str(), AP.mu_rel, AP.nonneg, AP.tol, AP.roi, AP.box,  pfix.c_str());
  }
  outpath_fname = (outpath_fname == "") ? CmdTools::creat_outpath(inpath_fname + "/" + buffer) : CmdTools::creat_outpath(outpath_fname + "/" + buffer);

  // The following is the reweighted iteration  
  ConvMsg* msg = new ConvMsg[AP.rwIter];
  clock_t t0 = clock();

  for(int n=0; n<AP.rwIter; n++) {
    if (verbose)
      cout<<"\nReweighted TV minimization iteration : "<<n<<endl;    

    msg[n] = SpAlgo::TVAL3(*P, *G, W, Y, Xr, //Nu, Lambda, 
			   (AP.epsilon<0) ? -1 : AP.epsilon*sqrt(conf.nbProj_total*conf.pixDet), // Noise level epsilon is scaled
			   AP.aniso, AP.nonneg, AP.mu, AP.beta, 
			   AP.tolInn, AP.tol, AP.tolGap, AP.maxIter, verbose);
    //msg[n].res *= Ynrml;

    // Update weighte
    if (AP.aniso)
      W = G->forward(Xr).abs();
    else
      W = SpAlgo::GTVNorm(G->forward(Xr), gradRank);
    double gmax = W.maxCoeff();
      
    Mask = (W <= AP.rwEpsilon * gmax / pow(2., n+1.)).select(1, ArrayXd::Zero(W.size())); // Edge mask
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

      if (AP.nbScale==1) {		// No simple way to save the mask for MS model 
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
  Xr *= AP.Ynrml;

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
  fout<<BI<<endl;
  fout<<endl;

  fout<<"[Projector Parameters]"<<endl;
  fout<<"Use strip-integral projector (stripint)="<<AP.stripint<<endl;
  //fout<<"Sinogram is normalzed (sgNrml)="<<AP.sgNrml<<endl;
  fout<<"Sinogram normalization value (Ynrml)="<<AP.Ynrml<<endl;
  fout<<endl;

  fout<<"[Algorithm Parameters]"<<endl;
  fout<<"Gradient lattice up-sampling factor (upSpl)="<<AP.upSpl<<endl;
  fout<<"Use anisotropic TV (aniso)="<<AP.aniso<<endl;
  fout<<"Use non negativity constraint (nonneg)="<<AP.nonneg<<endl;
  fout<<"Error level per detector bin (epsilon)="<<AP.epsilon<<endl;
  fout<<"Data fitting penalization (mu_rel)="<<AP.mu_rel<<endl;
  fout<<"Scaled data fitting penalization (mu)="<<AP.mu<<endl;
  fout<<"TVAL3: Gradient AL penalization (beta)="<<AP.beta<<endl;
  fout<<endl;

  fout<<"Global stopping tolerance (tol)="<<AP.tol<<endl;
  fout<<"TVAL3: Inner iteration stopping tolerance (tolInn)="<<AP.tolInn<<endl;
  fout<<"Maximum number of iterations (maxIter)="<<AP.maxIter<<endl;
  fout<<endl;

  fout<<"Reweighted TV: number of iterations (rwIter)="<<AP.rwIter<<endl;
  fout<<"Reweighted TV: weight mask threshold (rwEpsilon)="<<AP.rwEpsilon<<endl;
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
    fout<<endl<<"Reweighted TV iteration : "<<n<<endl;
    fout<<"Number of iterations="<<msg[n].niter<<endl;
    fout<<"Normalized residual |Ax-b|="<<msg[n].res<<endl;
    fout<<"TV norm="<<msg[n].norm<<endl;
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
