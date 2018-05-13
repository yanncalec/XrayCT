// Projector class functions

#include "Projector.hpp"

Projector::Projector(const AcqConfig &conf) :
  AcqConfig(conf), projset(ArrayXi::Zero(conf.nbProj_total))
{
  //  cout<<"Projector::Projector"<<endl;
  this->nbProj = this->nbProj_total;
  for (int n=0; n<this->nbProj_total; n++)
    this->projset[n] = n;

  this->maskDet.setConstant(nbProj*pixDet, true);
}

void Projector::set_nbProj(int nb)
{
  // Set the number of active projections to nb, which is included
  // between 1 and total number of projections. The
  // active projections are equally spaced from the initial
  // projections set.  If nb=0, the projection number is then
  // reset to this->nbProj_total.

  assert(nb>0);
  assert(nb<=nbProj_total);

  this->projset = ArrayXi::Zero(nb);
  double incr = this->nbProj_total * 1. / nb;
  //cout<<"incr :"<<incr<<endl;
  int tt = 0;
  for (int n=0; n<nb; n++)  {
    this->projset[n] = (int)floor(n * incr);
    //cout<<this->projset[n]<<", ";
    tt += this->projset[n];
  }
  if (this->projset[nb-1]>this->nbProj_total)
    this->projset[nb-1] = this->nbProj_total;

  this->set_projset(projset);
}

void Projector::reset_projset()
{
  //cout<<"Projector::reset_projset"<<endl;
  this->projset = ArrayXi::Zero(this->nbProj_total);
  for (int n=0; n<this->nbProj_total; n++)
    this->projset[n] = n;
  this->nbProj = this->nbProj_total;
  //  this->dimY = this->nbProj_total * this->pixDet;
  //  this->shapeY.y() = this->dimY;

  this->update_projector();
} 

void Projector::set_projset(const ArrayXi &projset)
{
  // Set the active projection subset (position of source) to
  // projset. If projset = NULL, the initial projection set is
  // restored.

  assert(projset.size() <= this->nbProj_total);
  this->nbProj = projset.size();
  this->projset = projset;

  // this->dimY = this->nbProj * this->pixDet;
  // this->shapeY.y() = this->dimY;

  this->update_projector();
}

void Projector::set_maskDet(const ArrayXb &M) 
{ 
  this->maskDet = M; 
  this->update_projector();
}

void Projector::reset_maskDet() 
{ 
  this->maskDet.setConstant(this->nbProj * this->pixDet, true); 
  this->update_projector();
}

ostream& operator<<(ostream& out, const Projector& P) {
  out<<*(AcqConfig *)&P;
  out<<"Active number of projections : "<<P.nbProj<<endl;
  out<<"Active projection subset : ";
  for (int n=0; n<P.nbProj; n++)
    out<<P.projset[n]<<", ";
  out<<endl;
  return out;
}


// Member functions for class LookupTableProj
GPUProjector::GPUProjector(const AcqConfig &conf)
  : Projector(conf), d_rSrc(0), d_pSrc(0), d_rDet(0)//, d_rtDet(0)
{
  this->init_projector();
}

GPUProjector::~GPUProjector()
{
  string fname = "GPUProjector::~GPUProjector";
  Tools::cudaSafe(cudaFree(this->d_rSrc), fname);
  Tools::cudaSafe(cudaFree(this->d_projset), fname);
  Tools::cudaSafe(cudaFree(this->d_pSrc), fname);
  Tools::cudaSafe(cudaFree(this->d_rDet), fname);
  Tools::cudaSafe(cudaFree(this->d_maskDet), fname);
  //  Tools::cudaSafe(cudaFree(this->d_rtDet), fname);
}

void GPUProjector::init_projector()
{ 
  string fname = "GPUProjector::GPUProjector";  
  size_t memsize_proj = this->nbProj * sizeof(double);
  size_t memsize_projset = this->nbProj * sizeof(int);
  size_t memsize_maskDet = this->nbProj * this->pixDet * sizeof(bool);

  Tools::cudaSafe(cudaMalloc((void **)&this->d_projset, memsize_projset), fname);
  Tools::cudaSafe(cudaMalloc((void **)&this->d_pSrc, memsize_proj), fname);
  Tools::cudaSafe(cudaMalloc((void **)&this->d_rSrc, memsize_proj), fname);
  //  Tools::cudaSafe(cudaMalloc((void **)&this->d_rtDet, memsize_proj), fname);
  Tools::cudaSafe(cudaMalloc((void **)&this->d_rDet, memsize_proj), fname);
  Tools::cudaSafe(cudaMalloc((void **)&this->d_maskDet, memsize_maskDet), fname);

  /* Upload data to device */
  Tools::cudaSafe(cudaMemcpy(this->d_projset, this->projset.data(), memsize_projset, cudaMemcpyHostToDevice), fname);
  Tools::cudaSafe(cudaMemcpy(this->d_pSrc, this->pSrc.data(), memsize_proj, cudaMemcpyHostToDevice), fname);
  Tools::cudaSafe(cudaMemcpy(this->d_rSrc, this->rSrc.data(), memsize_proj, cudaMemcpyHostToDevice), fname);
  Tools::cudaSafe(cudaMemcpy(this->d_rDet, this->rDet.data(), memsize_proj, cudaMemcpyHostToDevice), fname);
  Tools::cudaSafe(cudaMemcpy(this->d_maskDet, this->maskDet.data(), memsize_maskDet, cudaMemcpyHostToDevice), fname);
  //  Tools::cudaSafe(cudaMemcpy(this->d_rtDet, this->rtDet.data(), memsize_proj, cudaMemcpyHostToDevice), fname);
}
 
void GPUProjector::update_projector()
{
  string fname = "GPUProjector::update_projector()";
  Tools::cudaSafe(cudaFree(this->d_projset), fname);
  size_t memsize_projset = this->nbProj * sizeof(int);
  size_t memsize_maskDet = this->nbProj * this->pixDet * sizeof(bool);

  Tools::cudaSafe(cudaMalloc((void **)&this->d_projset, memsize_projset), fname);
  Tools::cudaSafe(cudaMemcpy(this->d_projset, this->projset.data(), memsize_projset, cudaMemcpyHostToDevice), fname);
  Tools::cudaSafe(cudaMemcpy(this->d_maskDet, this->maskDet.data(), memsize_maskDet, cudaMemcpyHostToDevice), fname);
}

// Member for class PixDrvProjector
void PixDrvProjector::PixDrvProjector_init()
{ 
  string fname = "PixDrvProjector::PixDrvProjector_init()";
  this->memsize_X = this->dimX * sizeof(double);
  this->memsize_Y = this->dimY * sizeof(double);
  
  Tools::cudaSafe(cudaMalloc((void **)&this->d_X, this->memsize_X), fname+" cudaMalloc 1");
  Tools::cudaSafe(cudaMalloc((void **)&this->d_Y, this->memsize_Y), fname+" cudaMalloc 2");  
}

PixDrvProjector::~PixDrvProjector()
{
  string fname = "PixDrvProjector::~PixDrvProjector";
  Tools::cudaSafe(cudaFree(this->d_X), fname);
  Tools::cudaSafe(cudaFree(this->d_Y), fname);  
}

void PixDrvProjector::_forward(const double *X, double *Y) 
{
  // upload X to device memory
  Tools::cudaSafe(cudaMemcpy(this->d_X, X, this->memsize_X, cudaMemcpyHostToDevice), "PixDrvProjector::_forward : cudaMemcpy() 1");
  Tools::cudaSafe(cudaMemset(this->d_Y, 0, this->memsize_Y),  "PixDrvProjector::_forward : cudaMemset 1");
  
  // Launch projection
  Tools::cudaSafe(PixDrvProjector_GPU(this->fanbeam,
				      this->nbProj,
				      this->d_projset,
				      this->d_pSrc,
				      this->d_rSrc,
				      this->d_rDet,
				      this->sizeDet, this->pixDet,
				      this->spObj,
				      this->dimObj.y(), this->dimObj.x(),
				      this->d_X,
				      this->d_Y,
				      true), "PixDrvProjector::_forward : PixDrvProjector_GPU");

  Tools::cudaSafe(cudaMemcpy(Y, this->d_Y, this->memsize_Y, cudaMemcpyDeviceToHost), "PixDrvProjector::_forward : cudaMemcpy 3");
}

void PixDrvProjector::_backward(const double *Y, double *X) 
{
  // upload X to device memory
  Tools::cudaSafe(cudaMemcpy(this->d_Y, Y, this->memsize_Y, cudaMemcpyHostToDevice), "PixDrvProjector::_backward : cudaMemcpy() 1");
  Tools::cudaSafe(cudaMemset(this->d_X, 0, this->memsize_X),  "PixDrvProjector::_backward : cudaMemset 1");
  
  // Launch projection
  Tools::cudaSafe(PixDrvProjector_GPU(this->fanbeam,
				      this->nbProj,
				      this->d_projset,
				      this->d_pSrc,
				      this->d_rSrc,
				      this->d_rDet,
				      this->sizeDet, this->pixDet,
				      this->spObj,
				      this->dimObj.y(), this->dimObj.x(),
				      this->d_Y,
				      this->d_X,
				      false), "PixDrvProjector::_backward : PixDrvProjector_GPU");

  Tools::cudaSafe(cudaMemcpy(X, this->d_X, this->memsize_X, cudaMemcpyDeviceToHost), "PixDrvProjector::_backward : cudaMemcpy 3");
}

