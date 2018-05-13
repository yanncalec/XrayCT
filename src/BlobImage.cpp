#include "BlobImage.hpp"

// Member functions for class BlobImage
BlobImage::BlobImage(const Blob *blob, const Grid *bgrid, const Array2d &sizeObj, double diamROI) 
  : nbScale(1), nbNode(bgrid->nbNode), sizeObj(sizeObj), diamROI(diamROI), scaling(1)
{
  // Single scale blob image  
  this->blob.push_back(blob);
  this->bgrid.push_back(bgrid);
}

BlobImage::BlobImage(const vector<const Blob *> &blob, const vector<const Grid *> &bgrid, const Array2d &sizeObj, double diamROI, double scaling) 
  : blob(blob), bgrid(bgrid), sizeObj(sizeObj), diamROI(diamROI), scaling(scaling)
{
  assert(blob.size() == bgrid.size());
  this->nbScale = blob.size();
  this->nbNode = 0;
  for (int n = 0; n<this->nbScale; n++)
    this->nbNode += bgrid[n]->nbNode;
}

// vector<ArrayXd> BlobImage::separate(const ArrayXd &X) const
// {
//   assert(X.size() == this->nbNode);
//   vector<ArrayXd > Y;
//   Y.resize(this->nbScale);
//   size_t offset = 0;

//   for (int n =0; n<this->nbScale; n++) {    
//     Y[n] = X.segment(offset, this->bgrid[n]->nbNode);
//     offset += this->bgrid[n]->nbNode;
//   }
//   return Y;
// }

// ArrayXd BlobImage::joint(const vector<ArrayXd> &Y) const
// {
//   assert(Y.size() == this->nbScale);
//   ArrayXd X(this->nbNode);
//   size_t offset = 0;
  
//   for (int n =0; n<this->nbScale; n++) {
//     assert(Y[n].size() == this->bgrid[n]->nbNode);
//     X.segment(offset, this->bgrid[n]->nbNode) = Y[n];
//     offset += this->bgrid[n]->nbNode;
//   }
//   return X;
// }

template<class T>
vector<T> BlobImage::separate(const T &X) const
{
  assert(X.size() == this->nbNode);
  vector<T> Y;
  Y.resize(this->nbScale);
  size_t offset = 0;

  for (int n =0; n<this->nbScale; n++) {    
    Y[n] = X.segment(offset, this->bgrid[n]->nbNode);
    offset += this->bgrid[n]->nbNode;
  }
  return Y;
}
template vector<ArrayXd> BlobImage::separate<ArrayXd>(const ArrayXd &X) const;
template vector<ArrayXb> BlobImage::separate<ArrayXb>(const ArrayXb &X) const;
template vector<ArrayXi> BlobImage::separate<ArrayXi>(const ArrayXi &X) const;

template<class T>
T BlobImage::joint(const vector<T> &Y) const
{
  assert(Y.size() == (size_t)this->nbScale);
  T X(this->nbNode);
  size_t offset = 0;
  
  for (int n=0; n<this->nbScale; n++) {
    assert(Y[n].size() == this->bgrid[n]->nbNode);
    X.segment(offset, this->bgrid[n]->nbNode) = Y[n];
    offset += this->bgrid[n]->nbNode;
  }
  return X;
}
template ArrayXd BlobImage::joint<ArrayXd>(const vector<ArrayXd> &Y) const;
template ArrayXi BlobImage::joint<ArrayXi>(const vector<ArrayXi> &Y) const;
template ArrayXb BlobImage::joint<ArrayXb>(const vector<ArrayXb> &Y) const;

// Interpolation from blob image to pixel image, for visualization only
ImageXXd BlobImage::blob2pixel(const ArrayXd &X, const Array2i &dimObj) const
{
  // Interpolation from a multi-blob image to a pixel image for screen display
  // X : blob coefficients

  //double pixSize = fmin(this->bgrid[0]->sizeObj.x()/dimObj.x(), this->bgrid[0]->sizeObj.y()/dimObj.y());
  double pixSize = fmin(sizeObj.x()/dimObj.x(), sizeObj.y()/dimObj.y());
  ArrayXd Y(dimObj.prod());
  ArrayXd Z(dimObj.prod());
  Y.setZero(); Z.setZero();
  //double *Y = new double[dimObj.prod()];

  size_t memsize_Y = dimObj.prod() * sizeof(double);
  double *d_X, *d_Y;
  Tools::cudaSafe(cudaMalloc((void **)&d_Y, memsize_Y), "BlobImage::blob2pixel() : cudaMalloc() 1");
  int offset = 0;

  for (int n =0; n<this->nbScale; n++) {
    size_t memsize_X = this->bgrid[n]->nbNode * sizeof(double);

    Tools::cudaSafe(cudaMalloc((void **)&d_X, memsize_X), "BlobImage::blob2pixel() : cudaMalloc() 2");
    Tools::cudaSafe(cudaMemcpy(d_X, X.data() + offset, memsize_X, cudaMemcpyHostToDevice), "BlobImage::blob2pixel() : cudaMemcpy() 1");
    offset += bgrid[n]->nbNode;
    Tools::cudaSafe(cudaMemset(d_Y, 0, memsize_Y), "BlobImage::blob2pixel() : cudaMemset() 1");

    Tools::cudaSafe(BlobInterpl_GPU(this->blob[n]->blobtype,
				    _Img_,
				    this->blob[n]->radius,
				    this->blob[n]->mul,
				    this->blob[n]->dil,
				    this->blob[n]->alpha, this->blob[n]->beta,
				    this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
				    this->bgrid[n]->splStep,
				    this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
				    this->bgrid[n]->d_mask,
				    this->bgrid[n]->nbNode,
				    0, -1,
				    pixSize,
				    dimObj.y(), dimObj.x(),
				    NULL,
				    dimObj.prod(),
				    d_X, 
				    d_Y), "BlobImage::blob2pixel() : BlobInterpl_GPU()");

    Tools::cudaSafe(cudaMemcpy(Z.data(), d_Y, memsize_Y, cudaMemcpyDeviceToHost), "BlobImage::blob2pixel() : cudaMemcpy() 2");
    Y += Z;
    cudaFree(d_X);
  }
  cudaFree(d_Y);
  return Map<ImageXXd> (Y.data(), dimObj.y(), dimObj.x()); // By default Eigen use column major  
  //return Y;
}

// Interpolation from blob image to multi pixel image, for visualization only
vector<ImageXXd> BlobImage::blob2multipixel(const ArrayXd &X, const Array2i &dimObj) const
{
  // Interpolation from a multi-blob image to a pixel image for screen display
  // X : blob coefficients

  // double pixSize = fmin(this->bgrid[0]->sizeObj.x()/dimObj.x(), this->bgrid[0]->sizeObj.y()/dimObj.y());
  double pixSize = fmin(sizeObj.x()/dimObj.x(), sizeObj.y()/dimObj.y());
  vector<ImageXXd> Y;
  Y.resize(this->nbScale);

  for (int n =0; n<this->nbScale; n++)
    Y[n].setZero(dimObj.y(), dimObj.x());

  size_t memsize_Y = dimObj.prod() * sizeof(double);
  double *d_X, *d_Y;
  Tools::cudaSafe(cudaMalloc((void **)&d_Y, memsize_Y), "BlobImage::blob2multipixel() : cudaMalloc() 1");
  int offset = 0;
  
  for (int n =0; n<this->nbScale; n++) {
    size_t memsize_X = this->bgrid[n]->nbNode * sizeof(double);

    Tools::cudaSafe(cudaMalloc((void **)&d_X, memsize_X), "BlobImage::blob2multipixel() : cudaMalloc() 2");
    Tools::cudaSafe(cudaMemcpy(d_X, X.data() + offset, memsize_X, cudaMemcpyHostToDevice), "BlobImage::blob2multipixel() : cudaMemcpy() 1");
    offset += bgrid[n]->nbNode;
    Tools::cudaSafe(cudaMemset(d_Y, 0, memsize_Y), "BlobImage::blob2multipixel() : cudaMemset() 1");

    Tools::cudaSafe(BlobInterpl_GPU(this->blob[n]->blobtype,
				    _Img_,
				    this->blob[n]->radius,
				    this->blob[n]->mul,
				    this->blob[n]->dil,
				    this->blob[n]->alpha, this->blob[n]->beta,
				    this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
				    this->bgrid[n]->splStep,
				    this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
				    this->bgrid[n]->d_mask,
				    this->bgrid[n]->nbNode,
				    0, -1,
				    pixSize,
				    dimObj.y(), dimObj.x(),
				    NULL,
				    dimObj.prod(),
				    d_X, 
				    d_Y), "BlobImage::blob2multipixel() : BlobInterpl_GPU()");

    Tools::cudaSafe(cudaMemcpy(Y[n].data(), d_Y, memsize_Y, cudaMemcpyDeviceToHost), "BlobImage::blob2multipixel() : cudaMemcpy() 2");

    cudaFree(d_X);
  }
  cudaFree(d_Y);
  return Y;
}

// Interpolation to gradient pixel image, returns Yx, Yy, for visualization only
vector<ImageXXd> BlobImage::blob2pixelgrad(const ArrayXd &X, const Array2i &dimObj) const
{
  // Interpolation from blob image to a gradient image for screen display
  // X : blob coefficients

  // double pixSize = fmin(this->bgrid[0]->sizeObj.x()/dimObj.x(), this->bgrid[0]->sizeObj.y()/dimObj.y());
  double pixSize = fmin(sizeObj.x()/dimObj.x(), sizeObj.y()/dimObj.y());
  ArrayXd Y(dimObj.prod() * 2);
  ArrayXd Z(dimObj.prod() * 2);
  Y.setZero(); Z.setZero();
  //double *Y = new double[dimObj.prod()];

  size_t memsize_Y = dimObj.prod() * sizeof(double) * 2;
  double *d_X, *d_Y;
  Tools::cudaSafe(cudaMalloc((void **)&d_Y, memsize_Y), "BlobImage::blob2pixelgrad() : cudaMalloc() 1");
  int offset = 0;

  for (int n =0; n<this->nbScale; n++) {
    size_t memsize_X = this->bgrid[n]->nbNode * sizeof(double);

    Tools::cudaSafe(cudaMalloc((void **)&d_X, memsize_X), "BlobImage::blob2pixelgrad() : cudaMalloc() 2");
    Tools::cudaSafe(cudaMemcpy(d_X, X.data() + offset, memsize_X, cudaMemcpyHostToDevice), "BlobImage::blob2pixelgrad() : cudaMemcpy() 1");
    offset += bgrid[n]->nbNode;
    Tools::cudaSafe(cudaMemset(d_Y, 0, memsize_Y), "BlobImage::blob2pixelgrad() : cudaMemset() 1");

    Tools::cudaSafe(BlobInterpl_GPU(this->blob[n]->blobtype,
				    _Grad_,
				    this->blob[n]->radius,
				    this->blob[n]->mul,
				    this->blob[n]->dil,
				    this->blob[n]->alpha, this->blob[n]->beta,
				    this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
				    this->bgrid[n]->splStep,
				    this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
				    this->bgrid[n]->d_mask,
				    this->bgrid[n]->nbNode,
				    0, -1,
				    pixSize,
				    dimObj.y(), dimObj.x(),
				    NULL,
				    dimObj.prod(),
				    d_X, 
				    d_Y), "BlobImage::blob2pixelgrad() : BlobInterpl_GPU()");

    Tools::cudaSafe(cudaMemcpy(Z.data(), d_Y, memsize_Y, cudaMemcpyDeviceToHost), "BlobImage::blob2pixelgrad() : cudaMemcpy() 2");
    Y += Z;
    cudaFree(d_X);
  }

  cudaFree(d_Y);

  //return Y;
  vector<ImageXXd> out;
  out.push_back(Map<ImageXXd> (Y.data(), dimObj.y(), dimObj.x())); // By default Eigen use column major 
  out.push_back(Map<ImageXXd> (Y.data()+dimObj.prod(), dimObj.y(), dimObj.x()));
  return out;
}

void BlobImage::interscale_product(const ArrayXd &X, ArrayXd &P) const
{
  assert(X.size() == this->get_nbNode());
  assert(P.size() == this->get_nbNode());

  P.head(this->bgrid[0]->nbNode) = X.head(this->bgrid[0]->nbNode);
  
  ArrayXd Y;
  double *d_X, *d_Y;
  size_t offset = 0;

  for (int n=1; n<this->nbScale; n++) {
    size_t memsize_X = this->bgrid[n-1]->nbNode * sizeof(double);
    Tools::cudaSafe(cudaMalloc((void **)&d_X, memsize_X), "BlobImage::interscale_product() : cudaMalloc() 1");
    Tools::cudaSafe(cudaMemcpy(d_X, X.data()+offset, memsize_X, cudaMemcpyHostToDevice), "BlobImage::interscale_product() : cudaMemcpy() 1");
    offset += this->bgrid[n-1]->nbNode;

    size_t memsize_Y = this->bgrid[n]->nbNode * sizeof(double);
    Y.setZero(this->bgrid[n]->nbNode);

    Tools::cudaSafe(cudaMalloc((void **)&d_Y, memsize_Y), "BlobImage::interscale_product() : cudaMalloc() 2");
    Tools::cudaSafe(cudaMemset(d_Y, 0, memsize_Y), "BlobImage::interscale_product() : cudaMemset() 2");

    // Interpolation to next scale
    Tools::cudaSafe(BlobInterpl_GPU(this->blob[n-1]->blobtype,
				    _Img_,
				    this->blob[n-1]->radius,
				    this->blob[n-1]->mul,
				    this->blob[n-1]->dil,
				    this->blob[n-1]->alpha, this->blob[n-1]->beta,
				    this->bgrid[n-1]->theta.x(), this->bgrid[n-1]->theta.y(),
				    this->bgrid[n-1]->splStep,
				    this->bgrid[n-1]->vshape.y(), this->bgrid[n-1]->vshape.x(),
				    this->bgrid[n-1]->d_mask,
				    this->bgrid[n-1]->nbNode,
				    this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
				    this->bgrid[n]->splStep,
				    this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
				    this->bgrid[n]->d_Idx,
				    this->bgrid[n]->nbNode,
				    d_X, 
				    d_Y), "BlobImage::interscale_product() : BlobInterpl_GPU()");

    Tools::cudaSafe(cudaMemcpy(Y.data(), d_Y, memsize_Y, cudaMemcpyDeviceToHost), "BlobImage::interscale_product() : cudaMemcpy() 2");
    Y /= Y.abs().maxCoeff();
    P.segment(offset, this->bgrid[n]->nbNode) = Y * X.segment(offset, this->bgrid[n]->nbNode);
      
    cudaFree(d_X);
    cudaFree(d_Y);
  }
}

vector<ArrayXd> BlobImage::interscale_product(const ArrayXd &X) const
{
  ArrayXd P(this->get_nbNode());
  this->interscale_product(X, P);
  vector<ArrayXd> vP = this->separate(P);  	
  return vP;
}

ArrayXi BlobImage::scalewise_prodmask(const ArrayXd &X, double beta, double spa) const
{
  assert(X.size() == this->get_nbNode());

  ArrayXd P(this->get_nbNode());
  this->interscale_product(X, P);

  ArrayXd Nr = this->scalewise_NApprx(P, beta, spa);
  return (Nr.abs() > 0).select(1, ArrayXi::Zero(Nr.size()));

  //return Mask;
}

ArrayXi BlobImage::prodmask(const ArrayXd &X, double spa) const
{
  assert(X.size() == this->get_nbNode());
  int nnz = (int)ceil(spa*X.size());

  ArrayXd P(this->get_nbNode());
  this->interscale_product(X, P);

  return SpAlgo::NApprx_support(P, nnz);
  //return (Nr.abs() > 0).select(1, ArrayXd::Zero(Nr.size()));
}

// ArrayXd BlobImage::prod_mask(const ArrayXd &X, double nz) const
// {
//   assert(X.size() == this->get_nbNode());

//   int dim = this->get_nbNode() - this->bgrid[0]->nbNode;
//   assert(nz > 0 and nz < 1);
//   int nterm = (int)ceil(nz * dim);

//   ArrayXd P(this->get_nbNode());
//   this->interscale_product(X, P);
  
//   ArrayXd Nr = SpAlgo::NApprx(P.tail(dim), nterm);
//   ArrayXd Mask = ArrayXd::Ones(P.size());
//   Mask.tail(dim) = (Nr.abs() > 0).select(1, ArrayXd::Zero(Nr.size()));

//   // ArrayXd Nr = SpAlgo::NApprx(P, nterm);
//   // ArrayXd Mask = ArrayXd::Ones(P.size());
//   // Mask = (Nr.abs() > 0).select(1, ArrayXd::Zero(Nr.size()));

//   return Mask;
// }

ArrayXd BlobImage::scalewise_NApprx(const ArrayXd &Xr, double beta, double spa) const
{
  // For each scale s>=1, keep only the biggest N_s coeffs
  // N_s ~ this->scaling^(-s) * spa, spa \in (0,1]
  // ie, if spa=1, scaling=2, then N_0 = 100%, N_1 = 50%, N_2 = 25%...
  // scale 0 is always preserved

  assert(Xr.size() == this->nbNode);
  beta = (beta==0) ? this->scaling : fabs(beta); // By default, follow the scale decaying rate
  vector<ArrayXd> XS = this->separate(Xr);

  for (size_t s = 0; s<XS.size(); s++) {
      ArrayXd XS_supp = (XS[s].abs()>0).select(1, ArrayXd::Zero(XS[s].size())); // coeff support of scale s
      int nnz = (int)ceil(XS[s].size() * spa / pow(beta, s)); 
      if (nnz < XS_supp.sum()) {
	ArrayXi Supp = SpAlgo::NApprx_support(XS[s], nnz);
	XS[s] = (Supp==0).select(0, XS[s]);
      }
    }
    return this->joint(XS);
}

void BlobImage::sparsity(const ArrayXd &X) const
{
  vector<ArrayXd> VX = this->separate(X);
  ArrayXi Idx;
  //double vmin, vmax, vmean, vstd;
  for (int n=0; n<this->nbScale; n++) {
    Idx = (VX[n].abs()>0).select(1, ArrayXi::Zero(VX[n].size()));    
    double std = (VX[n] - VX[n].mean()).matrix().norm() / sqrt(VX[n].size());
    printf("Scale %d : NNZ=%d,  Sparsity=%f,  Min=%1.5e,  Max=%1.5e,  Mean=%1.5e,  Std=%1.5e\n", n, Idx.sum(), Idx.sum() * 1. / Idx.size(), VX[n].minCoeff(), VX[n].maxCoeff(), VX[n].mean(), std);
  }
  Idx = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
  printf("All scales NNZ=%d,  Sparsity=%f\n", Idx.sum(), Idx.sum() * 1. / Idx.size());
}

ArrayXd BlobImage::reweighting_mask(const ArrayXd &Xr, double rwEpsilon)
{
  // vector<ArrayXd> Xs=this->separate(Xr);
  // Xs[0].setZero();    
  // ArrayXd Xt=this->joint(Xs);

  // ArrayXd toto = Xt.abs();
  // double gmax = toto.maxCoeff();
  // for (size_t m=0; m<toto.size(); m++)
  //   toto[m] = 1/(toto[m] + rwEpsilon * gmax);

  // Xs=this->separate(toto);
  // Xs[0].setZero();    
  // Xt = this->joint(Xs);

  ArrayXd toto = Xr.abs();
  double gmax = toto.maxCoeff();
  for (int m=0; m<toto.size(); m++)
    toto[m] = 1/(toto[m] + rwEpsilon * gmax);

  return toto/toto.maxCoeff();
}

// template<class T> T& operator<<(T& out, const BlobImage& BI)
// {
//   out<<"----Blob-Image----"<<endl;
//   out<<"Total number of active nodes : "<<BI.nbNode<<endl;
//   out<<"Total number of scales : "<<BI.nbScale<<endl;
//   for(int n=0; n<BI.nbScale; n++) {
//     out<<"Scale "<<n<<endl;
//     out<<*BI.blob[n];
//     out<<*BI.bgrid[n];
//   }
//   return out;
// }
// template ostream & operator<< <ostream>(ostream &out, const BlobImage &B);
// template ofstream & operator<< <ofstream>(ofstream &out, const BlobImage &B);

ostream & operator<<(ostream& out, const BlobImage& BI)
{
  out<<"----Blob-Image----"<<endl;
  out<<"Total number of active nodes : "<<BI.nbNode<<endl;
  out<<"Total number of scales : "<<BI.nbScale<<endl;
  out<<"Scaling factor : "<<BI.scaling<<endl;
  for(int n=0; n<BI.nbScale; n++) {
    out<<"---->Scale "<<n<<endl;
    out<<*BI.blob[n];
    out<<*BI.bgrid[n]<<endl;
  }
  return out;
}

ofstream & operator<<(ofstream& out, const BlobImage& BI)
{
  out<<"----Blob-Image----"<<endl;
  out<<"Total number of active nodes : "<<BI.nbNode<<endl;
  out<<"Total number of scales : "<<BI.nbScale<<endl;
  out<<"Scaling factor : "<<BI.scaling<<endl;
  for(int n=0; n<BI.nbScale; n++) {
    out<<"---->Scale "<<n<<endl;
    out<<*BI.blob[n];
    out<<*BI.bgrid[n]<<endl;
  }
  return out;
}
