#include "BlobInterpl.hpp"

// Member functions for BlobInterpl
void BlobInterpl::BlobInterpl_init()
{
  this->d_X.resize(nbScale);
  this->d_Y.resize(nbScale);
  this->memsize_X.setZero(nbScale);

  switch (linoptype) {
  case _Img_ : 
    this->set_shapeY(igrid->nbNode, 1);
    this->memsize_Y = igrid->nbNode * sizeof(double);
    this->linoptypeT = _ImgT_;
    this->set_linoptype(_Img_);
    break;
  case _Grad_ : 
    this->set_shapeY(igrid->nbNode, 2);
    this->memsize_Y = 2*igrid->nbNode * sizeof(double);
    this->linoptypeT = _GradT_;
    this->set_linoptype(_Grad_);
    break;
  case _Derv2_ : 
    this->set_shapeY(igrid->nbNode, 4);
    this->memsize_Y = 4*igrid->nbNode * sizeof(double);
    this->linoptypeT = _Derv2T_;
    this->set_linoptype(_Derv2_);
    break;
  default :
    cerr<<"Unknown interpolation type : "<<linoptype<<endl;
    exit(0);
  }

  for(int n=0; n<nbScale; n++) {
    this->memsize_X[n] = this->bgrid[n]->nbNode * sizeof(double); 
    Tools::cudaSafe(cudaMalloc((void **)&(this->d_X[n]), this->memsize_X[n]), "BlobInterpl::init() : cudaMalloc() 1");
    Tools::cudaSafe(cudaMalloc((void **)&(this->d_Y[n]), this->memsize_Y), "BlobInterpl::init() : cudaMalloc() 2");    
  }
  this->sY = new double[this->dimY];
}

BlobInterpl::~BlobInterpl()
{
  for (int n=0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaFree(this->d_X[n]), "BlobInterpl::~BlobInterpl() : cudaFree() 1");
    Tools::cudaSafe(cudaFree(this->d_Y[n]), "BlobInterpl::~BlobInterpl() : cudaFree() 2");
  }
  delete [] sY;
}

void BlobInterpl::_forward(const double *X, double *Z) 
{
  // Prepare memory
  int offset = 0;
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy(this->d_X[n], X + offset, memsize_X[n], cudaMemcpyHostToDevice), "BlobInterpl::_forward() : cudaMemcpy() 1");
    offset += this->bgrid[n]->nbNode;
    Tools::cudaSafe(cudaMemset(d_Y[n], 0, memsize_Y), "BlobInterpl::_forward() : cudaMemset() 1");
  }

  for (int n =0; n<this->nbScale; n++)
    Tools::cudaSafe(BlobInterpl_GPU(this->blob[n]->blobtype,
				    this->linoptype,
				    this->blob[n]->radius,
				    this->blob[n]->mul,
				    this->blob[n]->dil,
				    this->blob[n]->alpha, this->blob[n]->beta,
				    this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
				    this->bgrid[n]->splStep,
				    this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
				    this->bgrid[n]->d_mask,
				    this->bgrid[n]->nbNode,
				    this->igrid->theta.x(), this->igrid->theta.y(),
				    this->igrid->splStep,
				    this->igrid->vshape.y(), this->igrid->vshape.x(),
				    this->igrid->d_Idx,
				    this->igrid->nbNode,
				    d_X[n], 
				    d_Y[n]), "BlobInterpl::_forward() : BlobInterpl_GPU() 1");

  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy((n==0) ? Z : this->sY, this->d_Y[n], this->memsize_Y, cudaMemcpyDeviceToHost), "BlobInterpl::_forward() : cudaMemcpy() 2");
    if (n > 0)
      for (int d=0; d<this->dimY; d++)
	Z[d] += this->sY[d];
  }
}

void BlobInterpl::_backward(const double *Y, double *X) 
{
  // Prepare memory
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy(this->d_Y[n], Y, memsize_Y, cudaMemcpyHostToDevice), "BlobInterpl::_backward() : cudaMemcpy() 1");
    Tools::cudaSafe(cudaMemset(d_X[n], 0, memsize_X[n]), "BlobInterpl::_backward() : cudaMemset() 1");
  }

  for (int n =0; n<this->nbScale; n++)
    Tools::cudaSafe(BlobInterpl_GPU(this->blob[n]->blobtype,
				    this->linoptypeT,
				    this->blob[n]->radius,
				    this->blob[n]->mul,
				    this->blob[n]->dil,
				    this->blob[n]->alpha, this->blob[n]->beta,
				    this->igrid->theta.x(), this->igrid->theta.y(),
				    this->igrid->splStep,
				    this->igrid->vshape.y(), this->igrid->vshape.x(),
				    this->igrid->d_mask,
				    this->igrid->nbNode,
				    this->bgrid[n]->theta.x(), this->bgrid[n]->theta.y(),
				    this->bgrid[n]->splStep,
				    this->bgrid[n]->vshape.y(), this->bgrid[n]->vshape.x(),
				    this->bgrid[n]->d_Idx,
				    this->bgrid[n]->nbNode,
				    d_Y[n], 
				    d_X[n]), "BlobInterpl::_backward() : BlobInterpl_GPU() 1");

  int offset = 0;
  for (int n = 0; n<this->nbScale; n++) {
    Tools::cudaSafe(cudaMemcpy(X + offset, this->d_X[n], this->memsize_X[n], cudaMemcpyDeviceToHost), "BlobInterpl::_backward() : cudaMemcpy() 2");
    offset += this->bgrid[n]->nbNode;
  }
}

ostream& operator<<(ostream& out, const BlobInterpl & I) 
{
  out<<"----Blob Interpolation Operator----"<<endl;
  out<<*(BlobImage *)&I<<endl;
  out<<*(LinOp *) &I<<endl;
  return out;
}

