#include "Grid.hpp"
#include "Tools.hpp"

// Member functions of class Grid
Grid::Grid(const Array2d &sizeObj, double splStep, GridType gtype, double diamROI)
  : sizeObj(sizeObj), splStep(splStep), theta(Grid::type2theta(gtype)), gtype(gtype), diamROI(diamROI), gridname(Grid::type2str(gtype))
{
 // virtual rectangular shape
  this->vshape = this->grid_vshape(this->sizeObj, this->theta, this->splStep);

  this->init_grid();
};


Grid::Grid(const Array2i *vshape, double splStep, GridType gtype, double diamROI)
  : vshape(*vshape),  splStep(splStep), theta(Grid::type2theta(gtype)), gtype(gtype), diamROI(diamROI), gridname(Grid::type2str(gtype))
{
  // Determine the object size from sampling step and virtual shape.
  // the rectangular object is the biggest rectangular region included in the parallelogram
  // formed by vectors [1,0] and theta.
  assert(this->vshape.y() % 2 == 0);
  assert(this->vshape.x() % 2 == 0);

  // In the following caculation, 0.5 is added (enlarge sizeObj) to compensate the numerical error 
  // that might appeared in init_grid(when caculate the vshape of grid from sizeObj).
  this->sizeObj.x() = fabs(0.5 + this->vshape.x() - fabs(theta.x()) * this->vshape.y()) * splStep ;
  this->sizeObj.y() = fabs(theta.y()) * (this->vshape.y() + 0.5) * splStep ;
  // this->sizeObj.x() = fabs(vshape.x() - vshape.y() * theta.x()) * splStep ;
  // this->sizeObj.y() = fabs(vshape.y() * theta.y() + 0.5 * theta.y()) * splStep ;

  this->init_grid();
};

Grid::Grid(const Array2d &sizeObj, const Array2i &vshape, double splStep, GridType gtype, double diamROI)
  :  sizeObj(sizeObj), vshape(vshape), splStep(splStep), theta(Grid::type2theta(gtype)), gtype(gtype),  diamROI(diamROI), gridname(Grid::type2str(gtype))
{
  // Object size, virtual shape and sampling step are all given. 
  // the rectangular object is the biggest rectangular region included in the parallelogram
  // formed by vectors [1,0] and theta.
  assert(this->vshape.y() % 2 == 0);
  assert(this->vshape.x() % 2 == 0);
  assert(sizeObj.y() > 0);
  assert(sizeObj.x() > 0);

  this->init_grid();
}

Grid::~Grid() 
{
  Tools::cudaSafe(cudaFree(this->d_mask), "Grid::~Grid : cudaFree(d_mask)");
  Tools::cudaSafe(cudaFree(this->d_Idx), "Grid::~Grid : cudaFree(d_Idx)");
}

void Grid::init_grid()
{
  // Initialize grid, vshape and sizeObj must be already determined
  //cout<<"diamROI="<<diamROI<<endl;

  ArrayXi tmpIdx=ArrayXi::Zero(this->vshape.prod());
  this->nbNode = 0;
  this->mask = ArrayXi::Constant(this->vshape.prod(), -1);
  Array2d pos;
  // Calculate the index of active nodes
  for (int row=0; row<this->vshape.y(); row++)
    for (int col=0; col<this->vshape.x(); col++) {
      //pos = Tools::pix2posIV(row, col, this->theta, this->splStep, this->vshape);
      pos.x() = splStep * ((col-vshape.x()/2) + (row-vshape.y()/2) * theta.x());
      pos.y() = splStep * (row-vshape.y()/2) * theta.y();
      double rd = sqrt(pos.x()*pos.x() + pos.y()*pos.y());
      if (fabs(pos.x()) < this->sizeObj.x() /2 && fabs(pos.y()) < this->sizeObj.y() /2 && (diamROI==0 || rd<diamROI/2)) {
	int idx = row*this->vshape.x() + col;
	this->mask[idx] = this->nbNode;
	tmpIdx[this->nbNode++] = idx; // Row-major (C-order) : col + row * col_dimension
      }
    }

  this->Idx = tmpIdx.head(this->nbNode);

  this->upload_GPU();
}

void Grid::upload_GPU()
{
  // Upload to GPU
  size_t msize_mask = this->mask.size() * sizeof(int);
  Tools::cudaSafe(cudaMalloc((void **)&this->d_mask, msize_mask), "Grid::upload_GPU() : cudaMalloc()");
  Tools::cudaSafe(cudaMemcpy(this->d_mask, this->mask.data(), msize_mask, cudaMemcpyHostToDevice), "Grid::upload_GPU() : cudaMemcpy(d_mask)");

  size_t msize_Idx = this->Idx.size() * sizeof(int);
  Tools::cudaSafe(cudaMalloc((void **)&this->d_Idx, msize_Idx), "Grid::upload_GPU() : cudaMalloc()");
  Tools::cudaSafe(cudaMemcpy(this->d_Idx, this->Idx.data(), msize_Idx, cudaMemcpyHostToDevice), "Grid::upload_GPU() : cudaMemcpy(d_Idx)");
}

vector<ArrayXd> Grid::get_Nodes() const
{
  //int idx, row, col;
  //div_t q;
  ArrayXd Nodes_x, Nodes_y;

  Nodes_x.setZero(this->nbNode);
  Nodes_y.setZero(this->nbNode);
  for (int n=0; n<this->nbNode; n++) {
    // idx = this->Idx[n];
    // q = div(idx, this->vshape.x());
    // row = q.quot;  col = q.rem;
    Array2d pos = Tools::nodeIdx2pos(this->Idx[n], this->vshape, this->theta, this->splStep);
    Nodes_x[n] = pos.x();
    Nodes_y[n] = pos.y();
  }

  vector<ArrayXd> res;
  res.push_back(Nodes_x);
  res.push_back(Nodes_y);
  return res;
}

vector<ArrayXd> Grid::get_all_Nodes() const
{
  ArrayXd Nodes_x, Nodes_y;

  Nodes_x.setZero(this->vshape.prod());
  Nodes_y.setZero(this->vshape.prod());

  // Calculate the index of nodes
  for (int row=0; row<this->vshape.y(); row++)
    for (int col=0; col<this->vshape.x(); col++) {
      Array2d pos = Tools::node2pos(row, col, this->vshape, this->theta, this->splStep);      
      int n = row*this->vshape.x() + col;
      Nodes_x[n] = pos.x();
      Nodes_y[n] = pos.y();
    }

  vector<ArrayXd> res;
  res.push_back(Nodes_x);
  res.push_back(Nodes_y);
  return res;
}


Array2i Grid::grid_vshape(const Array2d &sizeObj, const Array2d &theta, double splStep) {
  // Find grid virtual shape (index range of parallelogram formed by [1,0] and theta)
  // given the rectangular sizeObj and theta, splStep
  // theta.x() = cos(theta), theta.y() = sin(theta)
  // dot(V, z) = [zx+theta.x() * zy, zy * theta.y()]
  double ly = sizeObj.y() / fabs(theta.y()); // parallelogram size in theta direction
  double lx = sizeObj.x() + ly * fabs(theta.x()); // parallelogram size in [1,0] direction
  int zy = (int)floor(ly / splStep / 2);
  int zx = (int)floor(lx / splStep / 2);
  return Array2i(2*zx, 2*zy);	// virtual shape must be even
}

void Grid::_embedding_forward(const double *X0, double *X) const {
  // Embed an array (dimension this->nbNode) to another (dimension
  // this->vshape.prod())

  for (int n = 0; n<this->vshape.prod(); n++) {
    if(this->mask[n] >= 0)
      X[n] = *(X0 + this->mask[n]);
    else
      X[n] = 0;
  }
}

void Grid::_embedding_backward(const double *X0, double *X) const {
  // Embed an array (dimension this->vshape.prod()) to another
  // (dimension this->nbNode)

  for (int n=0; n<this->nbNode; n++)
    X[n] = X0[this->Idx[n]];
}

void Grid::_embedding_complex_forward(const double *X0, fftw_complex *FX) const {
  // Embed an array (dimension this->nbNode) to another (dimension this->vshape.prod())

  for (int n = 0; n<this->vshape.prod(); n++) {
    if(this->mask[n] >= 0) {
      FX[n][0] = *(X0 + this->mask[n]);
      FX[n][1] = 0;
    }
    else {
      FX[n][0] = 0;
      FX[n][1] = 0;
    }
  }
}

void Grid::_embedding_complex_backward(const fftw_complex *FX, double *X) const {
  // Embed an array (dimension this->vshape.prod()) to another
  // (dimension this->nbNode)

  for (int n=0; n<this->nbNode; n++)
    X[n] = FX[this->Idx[n]][0];	// keep only the real part
}

ArrayXd Grid::embedding_forward(const ArrayXd &X0) const {
  // Embed an array (dimension this->nbNode) to another (dimension
  // this->vshape.prod())
  ArrayXd X(this->vshape.prod());
  X.setZero();

  this->_embedding_forward(X0.data(), X.data());
  return X;
}

ArrayXd Grid::embedding_backward(const ArrayXd &X0) const {
  // Embed an array (dimension this->vshape.prod()) to another
  // (dimension this->nbNode)

  ArrayXd X(this->nbNode);
  
  this->_embedding_backward(X0.data(), X.data());
  return X;
}

double Grid::determinant() const {
  return this->splStep * this->splStep * fabs(this->theta.y());
}

string Grid::type2str(GridType gtype)
{
  switch(gtype) {
  case _Hexagonal_ : 
    return "Hexagonal";
  case _Cartesian_ :
    return "Cartesian";
  default : 
    return "Unknown";
  }
}

Array2d Grid::type2theta(GridType gtype)
{
  switch(gtype) {
  case _Hexagonal_ :
    return Array2d(cos(M_PI/3), -sin(M_PI/3));
  case _Cartesian_ :
    return Array2d(0, -1);
  default :
    cerr<<"Unknown grid is not implemented!"<<endl;
    exit(1);
  }
}

void Grid::save(ofstream &out) const
{
  out.write(this->gridname.c_str(), sizeof(char)*this->gridname.size());
  out.write((char *)this->sizeObj.data(), sizeof(double)*this->sizeObj.size());
  out.write((char *)this->vshape.data(), sizeof(int)*this->vshape.size());
  out.write((char *)&this->splStep, sizeof(double));
}

// template<class T>
// T & operator<<(T &out, const Grid &grid)
// {
//   out<<"----Grid----"<<endl;
//   out<<"Grid type : "<<grid.gridname<<endl;
//   out<<"Generating vector : ("<<grid.theta.x()<<", "<<grid.theta.y()<<")"<<endl;
//   out<<"Sampling step : "<<grid.splStep<<endl;
//   out<<"Virtual shape of grid (R X C) : ("<<grid.vshape.y()<<", "<<grid.vshape.x()<<")"<<endl;
//   out<<"Object size (H X W) : ("<<grid.sizeObj.y()<<", "<<grid.sizeObj.x()<<")"<<endl;
//   out<<"Total number of nodes : "<<grid.vshape.prod()<<endl;
//   out<<"Number of active nodes : "<<grid.nbNode<<" ( "<<grid.nbNode*100./grid.vshape.prod()<<"% )"<<endl;
// }

// template ostream & operator<< <ostream>(ostream &out, const Grid &grid);
// template ofstream & operator<< <ofstream>(ofstream &out, const Grid &grid);

ostream & operator<<(ostream &out, const Grid &grid)
{
  out<<"----Grid----"<<endl;
  out<<"Grid type : "<<grid.gridname<<endl;
  out<<"Generating vector : ("<<grid.theta.x()<<", "<<grid.theta.y()<<")"<<endl;
  out<<"Sampling step : "<<grid.splStep<<endl;
  out<<"Virtual shape of grid (R X C) : ("<<grid.vshape.y()<<", "<<grid.vshape.x()<<")"<<endl;
  out<<"Object size (H X W) : ("<<grid.sizeObj.y()<<", "<<grid.sizeObj.x()<<")"<<endl;
  out<<"ROI diameter : "<<grid.diamROI<<endl;
  out<<"Total number of nodes : "<<grid.vshape.prod()<<endl;
  out<<"Number of active nodes : "<<grid.nbNode<<" ( "<<grid.nbNode*100./grid.vshape.prod()<<"% )"<<endl;

  return out;
}

ofstream & operator<<(ofstream &out, const Grid &grid)
{
  out<<"----Grid----"<<endl;
  out<<"Grid type : "<<grid.gridname<<endl;
  out<<"Generating vector : ("<<grid.theta.x()<<", "<<grid.theta.y()<<")"<<endl;
  out<<"Sampling step : "<<grid.splStep<<endl;
  out<<"Virtual shape of grid (R X C) : ("<<grid.vshape.y()<<", "<<grid.vshape.x()<<")"<<endl;
  out<<"Object size (H X W) : ("<<grid.sizeObj.y()<<", "<<grid.sizeObj.x()<<")"<<endl;
  out<<"ROI diameter : "<<grid.diamROI<<endl;
  out<<"Total number of nodes : "<<grid.vshape.prod()<<endl;
  out<<"Number of active nodes : "<<grid.nbNode<<" ( "<<grid.nbNode*100./grid.vshape.prod()<<"% )"<<endl;

  return out;
}


// ofstream & operator<<(ofstream &out, const Grid &grid)
// {
//   out.write(grid.gridname.c_str(), sizeof(char)*grid.gridname.size());
//   out.write((char *)grid.sizeObj.data(), sizeof(double)*grid.sizeObj.size());
//   out.write((char *)grid.vshape.data(), sizeof(int)*grid.vshape.size());
//   out.write((char *)&grid.splStep, sizeof(double));
//   return out;
// }

