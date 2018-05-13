#include "SpLinOp.hpp"

// Member for class PixGrad
PixGrad::PixGrad(const Array2i &dimObj, int mode)
  : dimObj(dimObj), LinOp(0, dimObj.prod(), _Grad_), mode(mode)
{
  this->set_shapeY(dimObj.prod(), 2);
  Xh.setZero(dimObj.y(), dimObj.x());
  Xv.setZero(dimObj.y(), dimObj.x());
}

void PixGrad::_forward(const double *X, double *Y)
{
  Map<ImageXXd> Dh(Y, dimObj.y(), dimObj.x());
  Map<ImageXXd> Dv(Y+dimObj.prod(), dimObj.y(), dimObj.x());
  //  Dh.setZero(); Dv.setZero();

  Map<ImageXXd> im(X, dimObj.y(), dimObj.x()); // Eigen column major : image matrix is transposed
  
  // Horizontal
  for(int cc = 0; cc<im.cols()-1; cc++)
    Dh.col(cc) = im.col(cc+1) - im.col(cc);

  if (mode == 0)
    Dh.col(im.cols()-1).setZero();
  else
    Dh.col(im.cols()-1) = im.col(0) - im.col(im.cols()-1);

  // Vertical
  for(int rr = 0; rr<im.rows()-1; rr++)
    Dv.row(rr) = im.row(rr+1) - im.row(rr);

  if (mode == 0)
    Dv.row(im.rows()-1).setZero();
  else
    Dv.row(im.rows()-1) = im.row(0) - im.row(im.rows()-1);
}

void PixGrad::_backward(const double *Y, double *X) 
{
  Map<ImageXXd> Dh(Y, dimObj.y(), dimObj.x());
  Map<ImageXXd> Dv(Y+dimObj.prod(), dimObj.y(), dimObj.x());

  ImageXXd Xh(dimObj.y(), dimObj.x());
  ImageXXd Xv(dimObj.y(), dimObj.x());

  Map<ImageXXd> im(X, dimObj.y(), dimObj.x());

  // Horizontal
  if (mode == 0)
    Xh.col(0).setZero();
  else
    Xh.col(0) = Dh.col(im.cols()-1) - Dh.col(0);

  for(int cc = 1; cc<im.cols(); cc++)
    Xh.col(cc) = Dh.col(cc-1) - Dh.col(cc);

  // Vertical
  if (mode == 0)
    Xv.row(0).setZero();
  else
    Xv.row(0) = Dv.row(im.rows()-1) - Dv.row(0);

  for(int rr = 1; rr<im.rows(); rr++)
    Xv.row(rr) = Dv.row(rr-1) - Dv.row(rr);

  im = Xh + Xv;
}

vector<ImageXXd> PixGrad::imgrad(const ImageXXd &im) 
{
  vector<ImageXXd> G; G.resize(2);
  
  ArrayXd toto = this->forward(Map<ArrayXd>(im.data(), im.size()));
  G[0] = Map<ImageXXd>(toto.data(), im.rows(), im.cols());
  G[1] = Map<ImageXXd>(toto.data()+im.size(), im.rows(), im.cols());

  return G;
}

ImageXXd PixGrad::imgradT(const vector<ImageXXd> &G) 
{
  ArrayXd toto(G[0].size()+G[1].size());
  toto.head(G[0].size()) = Map<ArrayXd>(G[0].data(), G[0].size());
  toto.tail(G[1].size()) = Map<ArrayXd>(G[1].data(), G[1].size());

  ArrayXd DD = this->backward(toto);
  return Map<ImageXXd>(DD.data(), G[0].rows(), G[0].cols());
}


// Member for GSL_Wavelet2D
GSL_Wavelet2D::GSL_Wavelet2D(const Array2i &dim, const string &wvlname, int order) 
  : LinOp(dim, 0) 
{
  this->nbScales = (int)ceil(log2(max(shapeY[0], shapeY[1]))); // k <= log2(max(shapeY[0], shapeY[1]))/2 <= k+1 = scales,
  this->gsl_dim = (int)pow(2, 1.*nbScales);
  // if gsl_dim differs from LinOp dimension, 
  // embedding/restriction operations will take place during the transform
  dimdiff = (shapeY[0] != gsl_dim or shapeY[1] != gsl_dim); 
  this->set_shapeX(gsl_dim, gsl_dim);  	// shapeX is coefficient dimension, must be squared, while shapeY is image dimension
  gsl_data = new double[gsl_dim*gsl_dim];
  //printf("nbScales:%d, gsl_dim:%d, dimdiff:%d\n", nbScales, gsl_dim, dimdiff);

  if(wvlname == "haar")
    this->wvl = gsl_wavelet_alloc(gsl_wavelet_haar, 2);
  else if(wvlname == "db") {
    assert(order % 2 == 0);
    assert(order>=4 and order<= 20);
    this->wvl = gsl_wavelet_alloc(gsl_wavelet_daubechies, order);
  }
  // case "bspline":
  //   wl = gsl_wavelet_alloc(gsl_wavelet_bspline, order);
  else {
    cerr<<"Unknown wavelet name.\n";
    exit(0);
  }

  //For two-dimensional transforms of n-by-n matrices it is
  //sufficient to allocate a workspace of size n, since the
  //transform operates on individual rows and columns
  wvl_ws = gsl_wavelet_workspace_alloc(this->gsl_dim);
}

GSL_Wavelet2D::~GSL_Wavelet2D()
{
  gsl_wavelet_free(this->wvl);
  gsl_wavelet_workspace_free(this->wvl_ws);
  delete [] gsl_data;
}

void GSL_Wavelet2D::transform(const double *X, double *Y, int dir)
{  
  int msg;

  if (dimdiff) {	  // Memory copy with embedding or restriction
    if(dir>0) { // forward, or synthesis (which is the inverse transform of gsl wvl): R*W^-1 
      for(int n=0; n<dimX; n++)	// Copy coefficient
	gsl_data[n] = X[n];    

      msg = gsl_wavelet2d_nstransform_inverse(wvl, gsl_data, gsl_dim, gsl_dim, gsl_dim, wvl_ws); // "backward" is gsl-reserved keyword

      for(int r=0; r<shapeY.y(); r++) // Restriction
	for(int c=0; c<shapeY.x(); c++)
	  Y[r*shapeY.x() + c] = gsl_data[r*shapeX.x() + c];
    }
    else{
      Tools::setZero(gsl_data, gsl_dim*gsl_dim);	// Embedding
      for(int r=0; r<shapeY.y(); r++)
	for(int c=0; c<shapeY.x(); c++)
	  gsl_data[r*shapeX.x() + c] = X[r*shapeY.x() + c];

      msg = gsl_wavelet2d_nstransform_forward(wvl, gsl_data, gsl_dim, gsl_dim, gsl_dim, wvl_ws); // "forward" is gsl-reserved keyword
      
      for(int n=0; n<dimX; n++)	// Copy coefficient
	Y[n] = gsl_data[n];
    }
  }
  else{				// No embedding or restriction
    for(int n=0; n<dimY; n++)
      Y[n] = X[n];    
    if(dir>0)
      msg = gsl_wavelet2d_nstransform_inverse(wvl, Y, gsl_dim, gsl_dim, gsl_dim, wvl_ws);
    else
      msg = gsl_wavelet2d_nstransform_forward(wvl, Y, gsl_dim, gsl_dim, gsl_dim, wvl_ws);
  }

  // gsl_matrix_const_view  view_in = gsl_matrix_const_view_array(X, this->gsl_dim, this->gsl_dim);
  // gsl_matrix_view view_out = gsl_matrix_view_array(Y, this->gsl_dim, this->gsl_dim);
  // gsl_matrix_memcpy(&view_out.matrix,&view_in.matrix);
  //gsl_wavelet2d_transform_matrix(wvl,&view_out.matrix, 1, wvl_ws);
  //int msg = gsl_wavelet2d_nstransform_matrix(wvl, &m_view_out.matrix, dir, wvl_ws);

  if(GSL_SUCCESS != msg){    
    fprintf(stderr,"GSL: %s\n", gsl_strerror(msg));
  }
}
