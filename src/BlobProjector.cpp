#include "BlobProjector.hpp"

GPUProjTable::GPUProjTable(const Blob &blob, bool stripint, size_t dimT)
  : d_T(0), stripint(stripint), dimT(dimT)
{
  //ArrayXd toto;
  if (stripint) {
    this->T = blob.StripTable(dimT);
    // for (size_t n=0; n<dimT; n++)
    //   this->T[n] = toto[n];
  }
  else {
    this->T = blob.AbelTable(dimT);
    // for (size_t n=0; n<dimT; n++)
    //   this->T[n] = toto[n];
  }
  size_t memsize_T = this->dimT * sizeof(double);

  Tools::cudaSafe(cudaMalloc((void **)&this->d_T, memsize_T), "GPUProjTable::GPUProjTable() : cudaMalloc() 1");
  Tools::cudaSafe(cudaMemcpy(this->d_T, this->T.data(), memsize_T, cudaMemcpyHostToDevice), "GPUProjTable::GPUProjTable() : cudaMemcpy() 1");
}

GPUProjTable::~GPUProjTable() 
{ 
  //delete[] this->T; 
  Tools::cudaSafe(cudaFree(this->d_T), "GPUProjTable::~GPUProjTable()"); 
}

ostream& operator<<(ostream& out, const GPUProjTable& LT)
{
  out<<"----GPU Integration Table----"<<endl;
  out<<"Strip-integral : "<<LT.stripint<<endl;
  out<<"Table size : "<<LT.dimT<<endl;
  out<<"Table first, middel and last element : "<<LT.T[0]<<", "<<LT.T[LT.dimT/2]<<", "<<LT.T[LT.dimT - 1]<<endl;
  return out;
}


// Member functions for class BlobProjector
//BlobProjector::BlobProjector(const AcqConfig &conf, const BlobImage *BI, bool dp, bool stripint, size_t dimT, bool sharetable)
BlobProjector::BlobProjector(const AcqConfig &conf, const BlobImage *BI, int tablemode, size_t dimT)
  : GPUProjector(conf), BlobImage(*BI), LinOp(Array2i(conf.pixDet, conf.nbProj_total), this->nbNode, _Proj_), tablemode(tablemode)
{   
  this->BlobProjector_init(); 
  this->LT.resize(this->nbScale);

  if (tablemode) {
    // Make Projection table    
    this->LT[0] = new GPUProjTable(*(BI->blob[0]), tablemode==2, dimT); // First scale (apprx) table
    if (this->nbScale > 1) {
      this->LT[1] = new GPUProjTable(*(BI->blob[1]), tablemode==2, dimT); // second scale (detail) table
    }    
    for (int n = 2; n<this->nbScale; n++)
      // all other detail tables are just the shared memory of second scale table
      this->LT[n] = this->LT[1];
  }

#ifdef _DEBUG_
  cout<<"BlobProjector::BlobProjector"<<endl;
#endif
}

void BlobProjector::BlobProjector_init()
{ 
  string fname = "BlobProjector::BlobProjector_init()";
  this->memsize_Y = this->dimY * sizeof(double);
  this->memsize_X.setZero(this->nbScale);

  this->d_X.resize(this->nbScale);
  this->d_Y.resize(this->nbScale);
  this->maskBlob.resize(this->nbScale);
  this->d_maskBlob.resize(this->nbScale);

  for (int n = 0; n<this->nbScale; n++) {  
    this->memsize_X[n] = this->bgrid[n]->nbNode * sizeof(double);
    //cout<<this->memsize_X[n]<<endl;
    Tools::cudaSafe(cudaMalloc((void **)&this->d_X[n], this->memsize_X[n]), fname+" cudaMalloc 1");
    Tools::cudaSafe(cudaMalloc((void **)&this->d_Y[n], this->memsize_Y), fname+" cudaMalloc 2");
    Tools::cudaSafe(cudaMalloc((void **)&this->d_maskBlob[n], this->bgrid[n]->nbNode*sizeof(bool)), fname+" cudaMalloc 3");
  }
  this->sY = new double[this->dimY];

  this->reset_maskBlob();
  //this->update_projector();
}

BlobProjector::~BlobProjector()
{
  string fname = "BlobProjector::~BlobProjector";
  // Tools::cudaSafe(cudaFree(this->d_T), fname);

  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaFree(this->d_X[n]), fname);
    Tools::cudaSafe(cudaFree(this->d_Y[n]), fname);
    Tools::cudaSafe(cudaFree(this->d_maskBlob[n]), fname);
  }
  delete [] this->sY;
}

void BlobProjector::_forward(const double *X, double *Z)
{
  // X is the input blob coeffs vector
  // It's the concatenation of X_0, X_1, .. X_N, each X_n 
  // is the blob coeff vector for a separate blob image.

  // initialize output device memory, necessary for projection
  for (int n = 0; n<this->nbScale; n++)
    Tools::cudaSafe(cudaMemset(this->d_Y[n], 0, this->memsize_Y),  "BlobProjector::_forward : cudaMemset 1");

  // upload X to device memory
  int offset = 0;
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy(this->d_X[n], X + offset, this->memsize_X[n], cudaMemcpyHostToDevice), "BlobProjector::_forward : cudaMemcpy() 1");
    offset += this->bgrid[n]->nbNode;
  }
  
  // Launch projection
  for (int n = 0; n<this->nbScale; n++)   
    Tools::cudaSafe(BlobProjector_GPU(this->blob[n]->blobtype,
				      this->blob[n]->radius, this->blob[n]->mul, this->blob[n]->dil,
				      this->blob[n]->alpha, this->blob[n]->beta,
				      this->fanbeam,
				      this->nbProj,
				      this->d_projset,
				      this->d_pSrc,
				      this->d_rSrc,
				      this->d_rDet,
				      this->sizeDet, this->pixDet,
				      (this->tablemode) ? this->LT[n]->d_T : 0, 
				      (this->tablemode) ? this->LT[n]->dimT : 0, 
				      (this->tablemode) ? this->LT[n]->stripint : false, 
				      this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
				      this->bgrid[n]->splStep,
				      this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
				      this->bgrid[n]->d_Idx,
				      this->bgrid[n]->nbNode,
				      this->d_X[n],
				      this->d_maskBlob[n],
				      this->d_Y[n],
				      this->d_maskDet,
				      true), "BlobProjector::_forward : BlobProjector_GPU");

  // Download output device memory to host
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy((n==0) ? Z : this->sY, this->d_Y[n], this->memsize_Y, cudaMemcpyDeviceToHost), "BlobProjector::_forward : cudaMemcpy 3");
    if (n > 0)
      for (int d=0; d<this->dimY; d++)
	Z[d] += this->sY[d];
  }
}

void BlobProjector::_backward(const double *Y, double *X)
{
  // upload Y to device memory
  for (int n = 0; n<this->nbScale; n++)
    Tools::cudaSafe(cudaMemcpy(this->d_Y[n], Y, this->memsize_Y, cudaMemcpyHostToDevice),  "BlobProjector::_backward : cudaMemcpy");

  // initialize output device memory, necessary for back-projection
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemset(this->d_X[n], 0, this->memsize_X[n]),  "BlobProjector::_backward : cudaMemset_dX");
    //    Tools::cudaSafe(cudaMemset(this->d_Y[n], 0, this->memsize_Y),  "BlobProjector::_backward : cudaMemset_dX");
  }

  // Launch projection
  for (int n = 0; n<this->nbScale; n++)
    Tools::cudaSafe(BlobProjector_GPU(this->blob[n]->blobtype,
				      this->blob[n]->radius, this->blob[n]->mul, this->blob[n]->dil,
				      this->blob[n]->alpha, this->blob[n]->beta,
				      this->fanbeam,
				      this->nbProj,
				      this->d_projset,
				      this->d_pSrc,
				      this->d_rSrc,
				      this->d_rDet,
				      this->sizeDet, this->pixDet,
				      (this->tablemode) ? this->LT[n]->d_T : 0, 
				      (this->tablemode) ? this->LT[n]->dimT : 0, 
				      (this->tablemode) ? this->LT[n]->stripint : false, 
				      this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
				      this->bgrid[n]->splStep,
				      this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
				      this->bgrid[n]->d_Idx,
				      this->bgrid[n]->nbNode,
				      this->d_Y[n],
				      this->d_maskDet,
				      this->d_X[n],
				      this->d_maskBlob[n],
				      false), "BlobProjector::_backward : BlobProjector_GPU");

  // Download output device memory to host
  int offset = 0;
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy(X+offset, this->d_X[n], this->memsize_X[n], cudaMemcpyDeviceToHost), "BlobProjector::_backward : cudaMemcpy 3");
    offset += this->bgrid[n]->nbNode;
  }
}

double BlobProjector::estimate_opNorm()
{
  // Estimation of spectral radius
  // |A|_2 <= sqrt(|A|_1 * |A|_inf), and |A|_1, |A|_inf norm can be evaluated easily

  ArrayXd Zr = this->row_lpnorm(1);
  ArrayXd Zc = this->col_lpnorm(1);

  // cout<<"BlobProjector::estimate_opNorm()"<<endl;
  // cout<<Zr.maxCoeff()<<endl;
  // cout<<Zc.maxCoeff()<<endl;
  return sqrt(Zr.maxCoeff() * Zc.maxCoeff());
}

ArrayXd BlobProjector::row_lpnorm(double lpnorm)
{
  ArrayXd Zr(dimY);
  // initialize output device memory, necessary for projection
  for (int n = 0; n<this->nbScale; n++)
    Tools::cudaSafe(cudaMemset(this->d_Y[n], 0, this->memsize_Y),  "BlobProjector::row_lpnorm : cudaMemset 1");

  for (int n = 0; n<this->nbScale; n++)
    Tools::cudaSafe(BlobProjector_NormEstimator_GPU(this->blob[n]->blobtype,
						    this->blob[n]->radius, this->blob[n]->mul, this->blob[n]->dil,
						    this->blob[n]->alpha, this->blob[n]->beta,
						    this->fanbeam,
						    this->nbProj,
						    this->d_projset,
						    this->d_pSrc,
						    this->d_rSrc,
						    this->d_rDet,
						    this->sizeDet, this->pixDet,
						    (this->tablemode) ? this->LT[n]->d_T : 0, 
						    (this->tablemode) ? this->LT[n]->dimT : 0, 
						    (this->tablemode) ? this->LT[n]->stripint : false, 
						    this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
						    this->bgrid[n]->splStep,
						    this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
						    this->bgrid[n]->d_Idx,
						    this->bgrid[n]->nbNode,
						    lpnorm,
						    true,
						    this->d_Y[n]), "BlobProjector::row_lpnorm : BlobProjector_GPU");

  // Download output device memory to host
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy((n==0) ? Zr.data() : this->sY, this->d_Y[n], this->memsize_Y, cudaMemcpyDeviceToHost), "BlobProjector::row_lpnorm : cudaMemcpy 1");
    if (n > 0)
      for (int d=0; d<this->dimY; d++)
	Zr[d] += this->sY[d];
  }
  if (lpnorm>0) { 
    for (int d=0; d<this->dimY; d++)
      Zr[d] = pow(Zr[d], 1./lpnorm);
  }

  return Zr;
}

ArrayXd BlobProjector::col_lpnorm(double lpnorm)
{
  ArrayXd Zc(dimX);

  // initialize output device memory, necessary for back-projection
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemset(this->d_X[n], 0, this->memsize_X[n]),  "BlobProjector::col_lpnorm : cudaMemset_dX");
  }

  for (int n = 0; n<this->nbScale; n++)
    Tools::cudaSafe(BlobProjector_NormEstimator_GPU(this->blob[n]->blobtype,
						    this->blob[n]->radius, this->blob[n]->mul, this->blob[n]->dil,
						    this->blob[n]->alpha, this->blob[n]->beta,
						    this->fanbeam,
						    this->nbProj,
						    this->d_projset,
						    this->d_pSrc,
						    this->d_rSrc,
						    this->d_rDet,
						    this->sizeDet, this->pixDet,
						    (this->tablemode) ? this->LT[n]->d_T : 0, 
						    (this->tablemode) ? this->LT[n]->dimT : 0, 
						    (this->tablemode) ? this->LT[n]->stripint : false, 
						    this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
						    this->bgrid[n]->splStep,
						    this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
						    this->bgrid[n]->d_Idx,
						    this->bgrid[n]->nbNode,
						    lpnorm,
						    false,
						    this->d_X[n]), "BlobProjector::col_lpnorm : BlobProjector_GPU");

  // Download output device memory to host
  int offset = 0;
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy(Zc.data()+offset, this->d_X[n], this->memsize_X[n], cudaMemcpyDeviceToHost), "BlobProjector::col_lpnorm : cudaMemcpy 1");
    offset += this->bgrid[n]->nbNode;
  }
  if (lpnorm>0) { 
    for (int d=0; d<this->dimX; d++)
      Zc[d] = pow(Zc[d], 1./lpnorm);
  }

  return Zc;
}

void BlobProjector::set_maskBlob(const vector<ArrayXb> &M)
{
  this->maskBlob = M;
  this->update_projector();  
}

void BlobProjector::reset_maskBlob() 
{ 
  for (int n=0; n<this->nbScale; n++)
    this->maskBlob[n].setConstant(this->bgrid[n]->nbNode, true);

  this->update_projector();
}

void BlobProjector::update_projector()
{
  //GPUProjector::update_projector();
  for (int n = 0; n<this->nbScale; n++) {  
    Tools::cudaSafe(cudaMemcpy(this->d_maskBlob[n], this->maskBlob[n].data(), this->bgrid[n]->nbNode * sizeof(bool), cudaMemcpyHostToDevice), "BlobProjector::update_projector");
  }
}


ostream& operator<<(ostream& out, const BlobProjector& P)
{
  out<<"----GPU Blob-Projector----"<<endl;
  out<<*(Projector *)&P;
  //out<<"Upper bound of spectral radius : "<<P.opNorm<<endl;

  out<<*(LinOp *)&P;
  if (P.tablemode) {
    for (int n=0; n<P.nbScale; n++)
      out<<*P.LT[n];
  }
  return out;
}
