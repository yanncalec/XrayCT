#include "BlobImage.hpp"
#include "BlobImageTools.hpp"

namespace BlobImageTools{
  #define EXTRADIUS 2.5

  BlobImage* SingleGaussPix(const Array2i &dimObj, double spObj, double cut_off_err, double fcut_off_err)
  {
    // cut_off_err : cut off error in space domain
    // fcut_off_err : cut off error in frequency domain

    Array2d sizeObj(dimObj.x() * spObj, dimObj.y() * spObj);
    double alpha = pow(M_PI / spObj / 2, 2) / log(2.);
    double radius = GaussBlob::eval_radius(alpha, cut_off_err); 
    double Radius = GaussBlob::eval_Radius(alpha, fcut_off_err); 
    return new BlobImage(new GaussBlob(radius, alpha), new Grid(&dimObj, spObj, _Cartesian_), sizeObj, 0);
  }

  BlobImage* SingleGauss(const Array2d &sizeObj, double diamROI, GridType gtype, double W, double cut_off_err, double fcut_off_err)
  {
    assert(gtype == _Hexagonal_ || gtype == _Cartesian_);
    double alpha = pow(M_PI * W / 2, 2) / log(2.);
    double r_Phi = GaussBlob::eval_radius(alpha, cut_off_err); 
    double R_Phi = GaussBlob::eval_Radius(alpha, fcut_off_err); 
    double h_Phi = (gtype == _Hexagonal_) ? 1./(sqrt(3) * R_Phi) : 1./(2 * R_Phi);

    // double splStep = (gtype == "hexagonal") ? M_PI/sqrt(-3 * log(fcut_off_err) * alpha) : M_PI/sqrt(-4 * log(fcut_off_err) * alpha);
    // Other methods for grid's sampling step
    // sqrt(log(2.) / alpha); 	// method 1
    // sqrt(2. / alpha); 	// method 2
    // M_PI/sqrt(3 * log(2.) * alpha); 	// method 3

    // Old version (0): r_Phi is added, which is wrong
    // Array2d vsizeObj = sizeObj + EXTRADIUS * r_Phi; // The virtual object size is a little bit bigger than the real one
    // double vdiamROI = diamROI +  EXTRADIUS * r_Phi;

    // New version:
    Array2d vsizeObj = sizeObj + EXTRADIUS * h_Phi; // The virtual object size is a little bit bigger than the real one
    double vdiamROI = diamROI +  EXTRADIUS * h_Phi;
    return new BlobImage(new GaussBlob(r_Phi, alpha), new Grid(vsizeObj, h_Phi, gtype, vdiamROI), sizeObj, diamROI);
  }

  BlobImage* MultiGaussDiffGauss(const Array2d &sizeObj, double diamROI, GridType gtype, double W, int nbScale, double cut_off_err, double fcut_off_err, bool tightframe)
  {
    if (nbScale == 1)
      return SingleGauss(sizeObj, diamROI, gtype, W, cut_off_err, fcut_off_err);

    assert(gtype == _Hexagonal_ || gtype == _Cartesian_);
    //double Alpha = 3 * pow(M_PI * W / pow(2., nbScale-1.), 2.) / log(4.); // This is the square apprx blob, the value is based on the position of \Psi_J pic
    double Alpha = pow(M_PI * W / pow(2., nbScale*1.), 2.) / log(2.); // This value is 1/6 of previous one, based on the sum of all scales frequency support, numerically more stable
    double alpha = Alpha * 2; // alpha of approximation gaussian blob

    double r_Phi = GaussBlob::eval_radius(alpha, cut_off_err); // Radius of \Phi
    double r_Psi = DiffGaussBlob::eval_radius(Alpha, cut_off_err); // Radius of \Psi

    double R_Phi = GaussBlob::eval_Radius(alpha, fcut_off_err); // Radius of \hat\Phi
    double R_Psi = DiffGaussBlob::eval_Radius(Alpha, fcut_off_err); // Radius of \hat\Psi

    double h_Phi =  (gtype == _Hexagonal_) ? 1./(sqrt(3) * R_Phi) :  1/(2 * R_Phi); // Sampling step of grid for \Phi
    double h_Psi =  (gtype == _Hexagonal_) ? 1./(sqrt(3) * R_Psi) :  1/(2 * R_Psi);

    vector <const Blob *> blob;
    vector <const Grid *> bgrid;
    blob.resize(nbScale);
    bgrid.resize(nbScale);

    // scale 0 : approximation blob image

    // Important remark on vsizeObj: The virtual object size
    // (vsizeObj) is a little bit bigger than the real one, since the
    // blobs outside object support can also contribute to the
    // image. The same principle applies also on multiscale grid.
    // Old version(0):
    // Array2d vsizeObj = sizeObj + EXTRADIUS * r_Phi;
    // double vdiamROI = diamROI +  EXTRADIUS * r_Phi;
    // New version(1):
    Array2d vsizeObj = sizeObj + EXTRADIUS * h_Phi;
    double vdiamROI = diamROI +  EXTRADIUS * h_Phi;

    bgrid[0] = new Grid(vsizeObj, h_Phi, gtype, vdiamROI);
    double cst = sqrt(2*alpha/M_PI);
    if (tightframe)
      blob[0] = new GaussBlob(r_Phi, alpha, sqrt(fabs(bgrid[0]->determinant()))*cst);
    else
      blob[0] = new GaussBlob(r_Phi, alpha, 1.);

    // Construction of detail blob images :
    double fc = 1;
    for (int n = 1; n<nbScale; n++) {
      // Old version
      // vsizeObj = sizeObj + EXTRADIUS * (r_Psi/fc);
      // vdiamROI = diamROI + EXTRADIUS * (r_Psi/fc);

      bgrid[n] = new Grid(vsizeObj, h_Psi / fc, gtype, vdiamROI);
      if (tightframe)
	blob[n] = new DiffGaussBlob(r_Psi/fc, Alpha, 100, sqrt(fabs(bgrid[n]->determinant()))*fc*fc, fc); // L1-normalization
      else
	blob[n] = new DiffGaussBlob(r_Psi/fc, Alpha, 100, fc, fc); // L2-normalization, in fact 1/h_Psi times the above term
      fc *= 2;
    }

    return new BlobImage(blob, bgrid, sizeObj, diamROI, 2.);
  }

  BlobImage* MultiGaussMexHat(const Array2d &sizeObj, double diamROI, GridType gtype, double W, int nbScale, double scaling, double cut_off_err, double fcut_off_err, bool tightframe)
  {
    if (nbScale == 1)
      return SingleGauss(sizeObj, diamROI, gtype, W, cut_off_err, fcut_off_err);

    assert(gtype == _Hexagonal_ || gtype == _Cartesian_);

    double L0 = W / pow(scaling, 1.*nbScale-2); // Bandwidth (the size of interval delimited by the two pics) of first scale mexhat
    double alpha_Psi = pow(M_PI*L0/2, 2.);  // parametre alpha of first scale mexhat, the pic of \hat\Psi is sqrt(\alpha)/pi = L0/2
    double R_Psi = MexHatBlob::eval_Radius(alpha_Psi, fcut_off_err); // Radius of \hat\Psi
    double r_Psi = MexHatBlob::eval_radius(alpha_Psi, cut_off_err);	 // Radius of \Psi
    double h_Psi = (gtype == _Hexagonal_) ? 1./(sqrt(3) * R_Psi) :  1/(2 * R_Psi); // Sampling step for first scale mexhat grid

    // We take simply the appproximation Gaussian blob parameter alpha in such a way that
    // its FWHM equals to L0/3
    double alpha_Phi = GaussBlob::FWHM2alpha(L0/3);
    double R_Phi = GaussBlob::eval_Radius(alpha_Phi, fcut_off_err);
    double r_Phi = GaussBlob::eval_radius(alpha_Phi, cut_off_err);
    double h_Phi =  (gtype == _Hexagonal_) ? 1./(sqrt(3) * R_Phi) :  1/(2 * R_Phi); // Sampling step of grid for \Phi

    vector <const Blob *> blob;
    vector <const Grid *> bgrid;
    blob.resize(nbScale);
    bgrid.resize(nbScale);

    // scale 0 : approximation blob image
    // Old version(0);
    // Array2d vsizeObj = sizeObj + EXTRADIUS * r_Phi; // The virtual object size is a little bit bigger than the real one
    // double vdiamROI = diamROI +  EXTRADIUS * r_Phi;

    Array2d vsizeObj = sizeObj + EXTRADIUS * h_Phi;
    double vdiamROI = diamROI +  EXTRADIUS * h_Phi;

    bgrid[0] = new Grid(vsizeObj, h_Phi, gtype, vdiamROI);
    double cst = alpha_Phi / exp(1.) / alpha_Psi; // multiplication factor for gaussian scale to approximate the partition of unity
    if (tightframe)
      blob[0] = new GaussBlob(r_Phi, alpha_Phi, sqrt(fabs(bgrid[0]->determinant())) * cst);
    else
      blob[0] = new GaussBlob(r_Phi, alpha_Phi, 1.);      

    // Construction of detail blob images :
    double fc = 1;
    for (int n = 1; n<nbScale; n++) {
      // Construct the grid;

      // Previous version: each scale has different vsizeObj/vdiamROI, therefore not a uniform covering of space
      // vsizeObj = sizeObj + EXTRADIUS * (r_Psi/fc); // The virtual object size is a little bit bigger than the real one
      // vdiamROI = diamROI + EXTRADIUS *  (r_Psi/fc);

      // New version: use the largest support size for all scales
      // vsizeObj = sizeObj + EXTRADIUS * h_Phi;
      // vdiamROI = diamROI + EXTRADIUS * h_Phi;

      // Newer version:
      // vsizeObj = bgrid[n-1]->sizeObj + EXTRADIUS * (r_Psi/fc);
      // vdiamROI = bgrid[n-1]->diamROI + EXTRADIUS * (r_Psi/fc);
            
      bgrid[n] = new Grid(vsizeObj, h_Psi / fc, gtype, vdiamROI);
      if (tightframe)
	blob[n] = new MexHatBlob(r_Psi/fc, alpha_Psi, sqrt(fabs(bgrid[n]->determinant()))*fc*fc, fc); // L1-normalization
      else
	blob[n] = new MexHatBlob(r_Psi/fc, alpha_Psi, fc, fc); // L2-normalization
      fc *= scaling;
    }

    return new BlobImage(blob, bgrid, sizeObj, diamROI, scaling);
  }

  BlobImage* MultiGaussD4Gauss(const Array2d &sizeObj, double diamROI, GridType gtype, double W, int nbScale, double scaling, double cut_off_err, double fcut_off_err, bool tightframe)
  {
    if (nbScale == 1)
      return SingleGauss(sizeObj, diamROI, gtype, W, cut_off_err, fcut_off_err);

    assert(gtype == _Hexagonal_ || gtype == _Cartesian_);

    double L0 = W / pow(scaling, 1.*nbScale-2); // Bandwidth (the size of interval delimited by the two picks) of first scale
    double alpha_Psi = pow(M_PI*L0/2, 2.) / 2;  // parametre alpha of first scale D4Gauss,  the pick of \hat\Psi is sqrt(2*\alpha)/pi = L0/2
    double R_Psi = D4GaussBlob::eval_Radius(alpha_Psi, fcut_off_err); // Radius of \hat\Psi
    double r_Psi = D4GaussBlob::eval_radius(alpha_Psi, cut_off_err);	 // Radius of \Psi
    double h_Psi = (gtype == _Hexagonal_) ? 1./(sqrt(3) * R_Psi) :  1/(2 * R_Psi); // Sampling step for first scale mexhat grid

    // We take simply the appproximation Gaussian blob parameter alpha in such a way that
    // its FWHM equals to L0/3
    double alpha_Phi = GaussBlob::FWHM2alpha(L0/3);
    double R_Phi = GaussBlob::eval_Radius(alpha_Phi, fcut_off_err);
    double r_Phi = GaussBlob::eval_radius(alpha_Phi, cut_off_err);
    double h_Phi =  (gtype == _Hexagonal_) ? 1./(sqrt(3) * R_Phi) :  1/(2 * R_Phi); // Sampling step of grid for \Phi

    vector <const Blob *> blob;
    vector <const Grid *> bgrid;
    blob.resize(nbScale);
    bgrid.resize(nbScale);

    // scale 0 : approximation blob image
    // Array2d vsizeObj = sizeObj + EXTRADIUS * r_Phi; // The virtual object size is a little bit bigger than the real one
    // double vdiamROI = diamROI +  EXTRADIUS * r_Phi;
    Array2d vsizeObj = sizeObj + EXTRADIUS * h_Phi; // The virtual object size is a little bit bigger than the real one
    double vdiamROI = diamROI +  EXTRADIUS * h_Phi;

    bgrid[0] = new Grid(vsizeObj, h_Phi, gtype, vdiamROI);
    double cst = alpha_Phi / exp(2.) / alpha_Psi; // multiplication factor for gaussian scale to approximate the partition of unity
    if (tightframe)
      blob[0] = new GaussBlob(r_Phi, alpha_Phi, sqrt(fabs(bgrid[0]->determinant())) * cst);
    else
      blob[0] = new GaussBlob(r_Phi, alpha_Phi, 1.);      

    // Construction of detail blob images :
    double fc = 1;
    for (int n = 1; n<nbScale; n++) {
      // Construct the grid;
      // vsizeObj = sizeObj + EXTRADIUS * (r_Psi/fc); // The virtual object size is a little bit bigger than the real one
      // vdiamROI = diamROI + EXTRADIUS * (r_Psi/fc);
      // New version: all scales have the same size
      // vsizeObj = sizeObj + EXTRADIUS * h_Psi; // The virtual object size is a little bit bigger than the real one
      // vdiamROI = diamROI + EXTRADIUS * h_Psi;

      bgrid[n] = new Grid(vsizeObj, h_Psi / fc, gtype, vdiamROI);
      if (tightframe)
	blob[n] = new MexHatBlob(r_Psi/fc, alpha_Psi, sqrt(fabs(bgrid[n]->determinant()))*fc*fc, fc); // L1-normalization
      else
	blob[n] = new D4GaussBlob(r_Psi/fc, alpha_Psi, fc, fc); // L2-normalization
      fc *= scaling;
    }

    return new BlobImage(blob, bgrid, sizeObj, diamROI, scaling);
  }

  int BlobImage_FOV_check(const BlobImage &BI, double FOV)
  {
    for (int n=0; n<BI.get_nbScale(); n++) {
      if ((BI.bgrid[n]->diamROI > FOV) and (BI.bgrid[n]->sizeObj.maxCoeff() > FOV/sqrt(2)))
	return n;
    }
    return -1;
  }

  void save(const BlobImage &BI, const string &fname)
  {
    char buffer[256];
    sprintf(buffer, "%s.dat", fname.c_str());

    ofstream out(buffer, ios::out | ios::binary);
    if (!out) {
      cout <<"Cannot open file : "<<buffer<<endl;
      exit(1);
    }
    //out.write((char *)&BI.get_nbScale(), sizeof(size_t));
    int nbScale = BI.get_nbScale();
    Array2d sizeObj = BI.get_sizeObj();
    double diamROI = BI.get_diamROI();
    double scaling = BI.get_scaling();
    out.write((char *)&nbScale, sizeof(int)); //This wont work : out<<BI.get_nbScale(); 
    out.write((char *)sizeObj.data(), sizeof(double)*2);
    out.write((char *)&diamROI, sizeof(double));
    out.write((char *)&scaling, sizeof(double));

    for (int n=0; n<nbScale; n++) {
      out.write((char *)&BI.blob[n]->radius, sizeof(double));
      out.write((char *)&BI.blob[n]->blobtype, sizeof(BlobType));
      out.write((char *)&BI.blob[n]->alpha, sizeof(double));
      out.write((char *)&BI.blob[n]->beta, sizeof(double));
      out.write((char *)&BI.blob[n]->mul, sizeof(double));
      out.write((char *)&BI.blob[n]->dil, sizeof(double));

      //out.write(BI.bgrid[n]->grid_name.c_str(), sizeof(char)*BI.bgrid[n]->grid_name.size());
      out.write((char *)BI.bgrid[n]->sizeObj.data(), sizeof(double) * 2);
      out.write((char *)BI.bgrid[n]->vshape.data(), sizeof(int) * 2);
      out.write((char *)&BI.bgrid[n]->splStep, sizeof(double));
      out.write((char *)&BI.bgrid[n]->gtype, sizeof(GridType));
      out.write((char *)&BI.bgrid[n]->diamROI, sizeof(double));
    }
    out.close();
  }

  BlobImage * load(const string &fname) 
  {    
    ifstream in(fname.data(), ios::in | ios::binary);
    if (!in) {
      cout <<"Cannot open file : "<<fname<<endl;
      exit(1);
    }
    int nbScale;
    in.read((char *)&nbScale, sizeof(int));

    Array2d sizeObj(0,0);
    in.read((char *)sizeObj.data(), sizeof(double) * 2);

    double diamROI;
    in.read((char *)&diamROI, sizeof(double));

    double scaling;
    in.read((char *)&scaling, sizeof(double));

    vector<const Blob *> blob; 
    vector<const Grid *> grid; 
    //blob.resize(nbScale); grid.resize(nbScale);

    for(int n=0; n<nbScale; n++) {
      double radius; BlobType blobtype; 
      double alpha, beta;
      double mul, dil;
      in.read((char *)&radius, sizeof(double));
      in.read((char *)&blobtype, sizeof(BlobType));
      in.read((char *)&alpha, sizeof(double));
      in.read((char *)&beta, sizeof(double));
      in.read((char *)&mul, sizeof(double));
      in.read((char *)&dil, sizeof(double));
      
      switch(blobtype) {
      case _GS_ : blob.push_back(new GaussBlob(radius, alpha, mul, dil)); break;
      case _DIFFGS_ : blob.push_back(new DiffGaussBlob(radius, alpha, int(beta), mul, dil)); break;
      case _MEXHAT_ : blob.push_back(new MexHatBlob(radius, alpha, mul, dil)); break;
      case _D4GS_ : blob.push_back(new D4GaussBlob(radius, alpha, mul, dil)); break;
      default : cerr<<"Unknown blob profile!"<<endl; exit(0);
      }

      Array2d sizeObj; Array2i vshape; double splStep; GridType gtype; double diamROI;
      in.read((char *)sizeObj.data(), sizeof(double) * 2);
      in.read((char *)vshape.data(), sizeof(int) * 2);
      in.read((char *)&splStep, sizeof(double));
      in.read((char *)&gtype, sizeof(GridType));
      in.read((char *)&diamROI, sizeof(double));
      grid.push_back(new Grid(sizeObj, vshape, splStep, gtype, diamROI));
    }
    in.close();
    return new BlobImage(blob, grid, sizeObj, diamROI, scaling);
  }
}
