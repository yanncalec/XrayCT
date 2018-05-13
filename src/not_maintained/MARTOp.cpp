// Member for class MARTOp
void MARTOp::MARTOp_init()
{ 
  string fname = "MARTOp::MARTOp_init()";
  // this->memsize_X = this->dimObj.prod() * sizeof(double);
  // this->memsize_Y = this->dimY * sizeof(double);
  
  Tools::cudaSafe(cudaMalloc((void **)&this->d_X, this->memsize_X), fname+" cudaMalloc 1");
  Tools::cudaSafe(cudaMalloc((void **)&this->d_Y, this->memsize_Y), fname+" cudaMalloc 2");  
}

MARTOp::~MARTOp()
{
  string fname = "MARTOp::~MARTOp";
  Tools::cudaSafe(cudaFree(this->d_X), fname);
  Tools::cudaSafe(cudaFree(this->d_Y), fname);  
}

void MARTOp::backward(const double *Y, double *X) 
{
  // upload X to device memory
  Tools::cudaSafe(cudaMemcpy(this->d_Y, Y, this->memsize_Y, cudaMemcpyHostToDevice), "MARTOp::_backward : cudaMemcpy() 1");
  Tools::cudaSafe(cudaMemcpy(this->d_X, X, this->memsize_X, cudaMemcpyHostToDevice), "MARTOp::_backward : cudaMemcpy() 2");
  //Tools::cudaSafe(cudaMemset(this->d_X, 0, this->memsize_X),  "MARTOp::_backward : cudaMemset 1");
  
  // Launch projection
  Tools::cudaSafe(MARTOp_GPU(this->fanbeam,
			     this->nbProj,
			     this->d_projset,
			     this->d_pSrc,
			     this->d_rSrc,
			     this->d_rDet,
			     this->sizeDet, this->pixDet,
			     this->spObj,
			     this->dimObj.y(), this->dimObj.x(),
			     this->d_Y, // input sinogram dat
			     this->d_X	// output pixel image vector
			     ), "MARTOp::_backward : MARTOp_GPU");

  Tools::cudaSafe(cudaMemcpy(X, this->d_X, this->memsize_X, cudaMemcpyDeviceToHost), "MARTOp::_backward : cudaMemcpy 3");
}

