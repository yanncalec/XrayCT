#include "Tools.hpp"
#include "LinOp.hpp"
#include "SpLinOp.hpp"

namespace Tools {

  template <class T> void setConstant(T* array, size_t N, T val) 
  {
    // Set an array to constant value
    for (size_t n=0; n<N; n++)
      array[n] = val;
  }
  template void setConstant<double> (double *array, size_t N, double val);
  template void setConstant<size_t> (size_t *array, size_t N, size_t val);
  template void setConstant<int> (int *array, size_t N, int val);

  template <class T> void setZero(T* array, size_t N) 
  {
    // Set an array to constant value
    for (size_t n=0; n<N; n++)
      array[n] = 0;
  }
  template void setZero<double> (double *array, size_t N);
  template void setZero<size_t> (size_t *array, size_t N);
  template void setZero<int> (int *array, size_t N);

  void setZeroComplex(fftw_complex * array, size_t N)
  {
    // Set an array to constant value
    for (size_t n=0; n<N; n++) {
      array[n][0] = 0;
      array[n][1] = 0;
    }
  }

  template <class T> void copyarray(const T *from, T *to, size_t N) 
  {
    // Copy an array
    for (size_t n=0; n<N; n++) {
      to[n] = from[n];
    }
  }
 
  template void copyarray<double> (const double *from, double *to, size_t N);
  template void copyarray<size_t> (const size_t *from, size_t *to, size_t N);

  template <class T> void truncarray(T *X, size_t N1, size_t N2)
  {
    // Truncation of array
    assert(N1<N2); 

    T *Y = new T[N2-N1+1];
    for (size_t n=N1; n<N2; n++) {
      Y[n-N1] = X[n];
    }
    X = Y;
  }

  template void truncarray<double>(double *X, size_t N1, size_t N2);
  template void truncarray<size_t>(size_t *X, size_t N1, size_t N2);

  double *float2double(const float *X, size_t N) 
  {
    // This is dangerous, *Y is a pointer to a local array. 
    // Calling this function from another function which returns Y will give wrong value ?
    double *Y = new double[N];
    for (int n=0; n<N; n++)
      Y[n] = (double) X[n];

    return Y;
  }

  void float2double(const float *X, double *Y, size_t N)
  {
    for (int n=0; n<N; n++)
      Y[n] = (double) X[n];
  }

  void double2float(const double *X, float *Y, size_t N)
  {
    for (int n=0; n<N; n++)
      Y[n] = (float) X[n];
  }

  ArrayXi subseq(int L0, int N, bool endpoint) 
  {
    assert(N<=L0);
    ArrayXd toto;
    if (endpoint) {
      toto = ArrayXd::LinSpaced(0, L0-1, N);
    }
    else {
      int sp = (int)floor(L0 * 1. / (N+1)); // spacing 
      toto = ArrayXd::LinSpaced(0, L0 - sp, N);
    }
    return toto.cast<int>();
  }

  ArrayXd sign(const ArrayXd &A)
  {
    // Return the sign of a double Eigen array
    return (A>=0).select(1, ArrayXd::Constant(A.size(), -1));
  }

  // ArrayXXd rowwise_prod(const ArrayXXd &A, const ArrayXd &v)
  // {
  //   // Termwise multiplication of each row of A by v
  //   assert(A.cols() == v.size());
  //   ArrayXXd B = A;
  //   for (size_t c=0; c<A.cols(); c++)
  //     B.col(c) *= v[c];
  //   return B;
  // }

  // ArrayXXd colwise_prod(const ArrayXXd &A, const ArrayXd &v)
  // {
  //   // Termwise multiplication of each column of A by v
  //   assert(A.rows() == v.size());
  //   ArrayXXd B = A;
  //   for (size_t c=0; c<A.rows(); c++)
  //     B.row(c) *= v[c];
  //   return B;
  // }

  // void mat_diag_inplace(ArrayXXd &A, const ArrayXd &D)
  // {
  //   // Multiplication of A*D, with D diagonal matrix
  //   // This multiplies n-th column of A by D(n)
  //   assert(A.cols() == D.size());
  //   for (size_t n=0; n<A.cols(); n++)
  //     A.col(n) *= D[n];
  // }

  // void diag_mat_inplace(const ArrayXd &D, ArrayXXd &A)
  // {
  //   // Multiplication of D*A, with D diagonal matrix
  //   // This multiplies n-th row of A by D(n)
  //   assert(A.rows() == D.size());
  //   for (size_t n=0; n<A.rows(); n++)
  //     A.row(n) *= D[n];
  // }

  // ArrayXd rowwise_normalize_inplace(ArrayXXd &A)
  // {
  //   // Normalize each row of A and return the norm
  //   ArrayXd N = (A*A).rowwise().sum().sqrt();

  //   for(size_t n=0; n<A.rows(); n++){
  //     if (N[n]>1e-10)
  // 	A.row(n) /= N[n];
  //   }
  //   return N;
  // }

  ArrayXd cumsum(const ArrayXd &A)
  {
    ArrayXd B = A;

    for (size_t n=1; n<A.size(); n++)
      B[n] += B[n-1];
    return B;
  }

  int qsort_comp_func_ascent(const void *a,const void *b)
  {
    double *arg1 = (double *) a;
    double *arg2 = (double *) b;
    if( *arg1 < *arg2 ) return -1;
    else if( *arg1 == *arg2 ) return 0;
    else return 1;
  }

  int qsort_comp_func_descent(const void *a,const void *b)
  {
    double *arg1 = (double *) a;
    double *arg2 = (double *) b;
    if( *arg1 > *arg2 ) return -1;
    else if( *arg1 == *arg2 ) return 0;
    else return 1;
  }

  size_t find_idx_eq(const size_t *A, size_t M, size_t x, size_t *Idx)
  {
    size_t N=0;
    for (size_t n=0; n<M; n++)
      if(A[n]==x)
  	Idx[N++] = n;
    return N;
  }

  size_t find_idx_eq(const ArrayXu& A, size_t x, ArrayXu &Idx)
  {
    size_t N=0;
    for (size_t n=0; n<A.size(); n++)
      if(A[n]==x)
  	Idx[N++] = n;
    return N;
  }

  size_t find_idx_leq(const ArrayXu& A, size_t x, ArrayXu &Idx)
  {
    size_t N=0;
    for (size_t n=0; n<A.size(); n++)
      if(A[n]<=x)
  	Idx[N++] = n;
    return N;
  }
  
  bool isequal(double a, double b)
  {
    return (islessequal(a,b) and islessequal(b,a));
  }
  
  double relative_error(double a, double b){
    return fabs(a-b)/b;
  }

  Array2d nodeIdx2pos(int nodeIdx, const Array2i &vshape, const Array2d &theta, double splStep)
  {
    int rr = (int)floor(nodeIdx * 1. / vshape.x());
    int cc = nodeIdx - rr * vshape.x();
    
    return Array2d(splStep * (cc - vshape.x()/2 + (rr - vshape.y()/2) * theta.x()), splStep * (rr - vshape.y()/2) * theta.y());
  }

  Array2d node2pos(int row, int col, const Array2d &theta, double splStep)
  {
    // Calculate the position of a given node wrt the grid vector v1=[1,0]'
    // V = [v1, theta], sampling step
    return Array2d(splStep * (col + row * theta.x()), splStep * row * theta.y());
  }

  Array2d node2pos(int row, int col, const Array2i &vshape, const Array2d &theta, double splStep)
  {
    return node2pos(row-vshape.y()/2, col-vshape.x()/2, theta, splStep);
  }

  Array2d pix2pos(int row, int col, const Array2i &dimObj, double pixSize)
  {
    /* Conversion from pixel to cartesian position, image is centered, with left-up corner the (0,0) pixel */
    return Array2d((col - dimObj.x()/2)* pixSize, (dimObj.y()/2 - row)* pixSize);
  }

  Array2d pix2pos(int row, int col, size_t rows, size_t cols, double pixSize)
  {
    /* Conversion from pixel to cartesian position, image is centered, with left-up corner the (0,0) pixel */
    return Array2d((col - cols/2)* pixSize, (rows/2 - row)* pixSize);
  }

  Array2i pos2pix(const Array2d &pos, double spObj, const Array2d &sizeObj)
  /* Conversion from cartesian position to pixel, image is centered, with left-up corner the (0,0) pixel */
  {
    Array2i dimObj;		/* Object's resolution in pixel */
    dimObj.y() = (int)floor(sizeObj.y() / spObj); /* sizeObj[0] is the height(y-axis) */
    dimObj.x() = (int)floor(sizeObj.x() / spObj); /* sizeObj[0] is the width(x-axis) */

    int cc =  (int)trunc((pos.x() + sizeObj.x()/2)/spObj);
    int rr = -(int)trunc((pos.y() - sizeObj.y()/2)/spObj);

    rr=(rr<0)?0:rr;
    rr=(rr>dimObj.y()-1)?(dimObj.y()-1):rr;
    cc=(cc<0)?0:cc;
    cc=(cc>dimObj.x()-1)?(dimObj.x()-1):cc;

    return Array2i(cc, rr);
  }

  Array2i pos2pix(const Array2d &pos, double spObj, size_t rows, size_t cols)
  /* Conversion from cartesian position to pixel, image is centered, with left-up corner the (0,0) pixel */
  {
    Array2d sizeObj(cols * spObj, rows * spObj);

    int cc =  (int)trunc((pos.x() + sizeObj.x()/2)/spObj);
    int rr = -(int)trunc((pos.y() - sizeObj.y()/2)/spObj);

    rr=(rr<0)?0:rr;
    rr=(rr>rows-1)?(rows-1):rr;
    cc=(cc<0)?0:cc;
    cc=(cc>cols-1)?(cols-1):cc;

    return Array2i(cc, rr);
  }

  Array2i pos2pix(const Array2d &pos, double spObj, const Array2i &dimObj) 
  {
    return pos2pix(pos, spObj, dimObj.y(), dimObj.x());
  }

  inline double modulo(double a, double b) 
  // Return the remainder of a - c*b, with c the minmum divider (abslute value) 
  {
    return a - trunc(a/b) * b;
  }

  inline double dist_gaussian(double a, double b, double sigma) 
  // Return the probability of a zero-mean gaussian r.v. in interval [a,b] : P(X \in (a,b)) sigma>0
  {
    return fabs(erf(b/sqrt(2)/sigma) - erf(a/sqrt(2)/sigma))/2;
  }

  template<class T>
  double SNR(const T &X, const T &Y)
  {
    // Calculate the Signal to Noise Ratio(SNR) of
    // noisy signal X compared to noiseless Y.
    // standard version X and Y are NOT normalized.
    // SNR(X,Y) := 10 * log10(e_s / e_n)    
    // with e_s and e_n the enery of signal and noise
    assert(X.size() == Y.size());

    // Energy of noise 
    T Noise = X - Y;
    double mN = Noise.mean();
    double eN = (Noise - mN).matrix().squaredNorm();

    // Energy of signal
    double mS = Y.mean();
    double eS = (Y - mS).matrix().squaredNorm();

    return 10 * log10(eS / eN);
    
  }

  template double SNR<ArrayXd>(const ArrayXd &X, const ArrayXd &Y);
  template double SNR<ImageXXd>(const ImageXXd &X, const ImageXXd &Y);

  template<class T>
  double Corr(const T &X, const T &Y)
  {
    // Calculate the correlation between signal X and Y.
    // X and Y are normalized.
    // corr(X,Y) := dot(X-mean(X), Y-mean(Y))/norm(X-mean(X))/norm(Y-mean(Y))
    assert(X.size() == Y.size());

    T X0 = X - X.mean();
    T Y0 = Y - Y.mean();
    X0 /= X0.matrix().norm();
    Y0 /= Y0.matrix().norm();
    return (X0*Y0).sum();
  }
  template double Corr<ArrayXd>(const ArrayXd &X, const ArrayXd &Y);
  template double Corr<ImageXXd>(const ImageXXd &X, const ImageXXd &Y);

  template<class T>
  double UQI(const T &X, const T &Y)
  {
    assert(X.size() == Y.size());
    double mX = X.mean(); 
    double mY = Y.mean();    
    double vX = (X-mX).matrix().squaredNorm();
    double vY = (Y-mY).matrix().squaredNorm();
    double covXY = ((X-mX)*(Y-mY)).sum();
    // printf("mX=%f, mY=%f, vX=%f, vY=%f, covXY=%f\n", mX, mY, vX, vY, covXY);
    // printf("q1=%f, q2=%f\n",(2*mX*mY)/(mX*mX + mY*mY), (2*covXY) / (vX + vY));
    return (2*mX*mY)/(mX*mX + mY*mY) * (2*covXY) / (vX + vY);
  }
  template double UQI<ArrayXd>(const ArrayXd &X, const ArrayXd &Y);
  template double UQI<ImageXXd>(const ImageXXd &X, const ImageXXd &Y);

  double StreakIndex(const ArrayXd &X, const ArrayXd &im, const Array2i dim)
  {
    assert(X.size() == im.size());
    PixGrad *G = new PixGrad(dim);
    ArrayXd S = X-im;

    ArrayXd GS = G->forward(S);
    Map<ArrayXd> Dh(GS.data(), im.size());
    Map<ArrayXd> Dv(GS.data()+im.size(), im.size());
    return (Dh*Dh + Dv*Dv).sqrt().sum() / dim.prod();
  }

  double StreakIndex(const ImageXXd &X, const ImageXXd &im)
  {
    assert(X.cols() == im.cols());
    assert(X.rows() == im.rows());

    Array2i dim(im.cols(), im.rows());
    PixGrad *G = new PixGrad(dim);

    ImageXXd toto = X-im;
    Map<ArrayXd> S(toto.data(), toto.size());
    ArrayXd GS = G->forward(S);
    Map<ArrayXd> Dh(GS.data(), im.size());
    Map<ArrayXd> Dv(GS.data()+im.size(), im.size());

    return (Dh*Dh + Dv*Dv).sqrt().sum() / dim.prod();
  }

  ArrayXu Photon(const ArrayXd &X, size_t phs)
  {    
    gsl_rng_env_setup();
    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng *r = gsl_rng_alloc(T);

    ArrayXu Y = ArrayXu::Zero(X.size());

    for (size_t n=0; n<X.size(); n++)
      Y[n] = gsl_ran_poisson(r, exp(-X[n]) * phs);

    gsl_rng_free(r);
    return Y;
  }

  ArrayXd LogPhoton(const ArrayXu &I, size_t phs)
  {
    // SG = LogPhoton(I, phs)
    // Convert photon data to noisy sinogram.
    // I : Photon counting sinogram
    // phs : Source photon intensity
    // SG : converted log-sinogram

    // By Beer-Lambert law : B ~ phs * exp(-Ax)
    // When Ax is too large, B can take 0. We treat
    // this case by truncation.

    ArrayXu nI = (I==0).select(1, I);
    ArrayXd Y = ArrayXd::Zero(I.size());
    double lphs = log((double)phs);

    for(size_t n=0; n<I.size(); n++)
      Y[n] = lphs - log((double)nI[n]);

    return Y;
  }

  ArrayXd poisson_data(const ArrayXd &Y, size_t phs, Array3d &info)
  {
    // B, (sigma, epsilon, snr) = poisson_data(Y, phs)
    // Simulate photon generation and convert the photon data to log-photon data.

    // Inputs :
    // Y : noiseless sinogram vector, must be positive and non all zero
    // phs : photon intensity at source
    
    // Outputs :
    // B : noisy sinogram in image form
    // sigma : noise's standard deviation
    // epsilon : sigma * sqrt(B.size)
    // snr : signal-to-noise ratio of noisy sinogram B
    double SC = Y.maxCoeff();
    ArrayXu I = Photon(Y/SC, phs);
    ArrayXd B = LogPhoton(I, phs) * SC;

    double energy = (B - Y).matrix().squaredNorm();
    double sigma = sqrt(energy / Y.size());
    double snr = SNR(B, Y);
    // SC = np.max(Y)                         # Scale factor
    // I = Photon(Y/SC, phs)               # Y is scaled in order to prevent numerical problems
    // B = LogPhoton(I, phs) * SC
    // epsilon = norm(B - Y)
    // sigma = epsilon / np.sqrt(Y.size)
    // snr = 20 * np.log10(norm(Y-np.mean(Y)) / norm(B-Y))
    info = Array3d(energy, sigma, snr);
    return B; 
  }

  ArrayXd gaussian_data(const ArrayXd &Y, double snr, Array3d &info)
  {
    // B, (sigma, epsilon) = gaussian_data(Y, snr)
    // Simulate gaussian data of given SNR level (db).
    
    // Outputs :
    // B : noisy sinogram in image form
    // sigma : noise's standard deviation
    // epsilon : sigma * sqrt(B.size)
    
    // Y = P.forward(X.flatten()) # Noiseless sinogram
    //epsilon = norm(Y - np.mean(Y)) / np.sqrt(10**(snr/10) - 1)
    double epsilon = (Y - Y.mean()).matrix().norm() / sqrt(pow(10,(snr/10)));
    double sigma = epsilon / sqrt(Y.size());

    gsl_rng_env_setup();
    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng *r = gsl_rng_alloc(T);

    ArrayXd B = ArrayXd::Zero(Y.size());

    for (size_t n=0; n<Y.size(); n++)
      B[n] = Y[n] + gsl_ran_gaussian(r, sigma);

    gsl_rng_free(r);
    
    info = Array3d (epsilon, sigma, SNR(B, Y));
    return B;
  }

  double sg_noiselevel(const ArrayXd Y, int nbProj, int pixDet)
  {
    // This is a reasonable estimation only for parallel beam
    assert(Y.size() == nbProj * pixDet);
    ArrayXd Sk(nbProj);
    for (int p=0; p<nbProj; p++)
      Sk[p] = Y.segment(p*pixDet, pixDet).mean();
    // Map<ArrayXXd> SG(Y.data(), pixDet, nbProj); // Sinogram
    // ArrayXd Sk = SG.colwise().sum() / pixDet;
    // cout<<Sk.size()<<endl;
    double mS = Sk.mean();
    return sqrt((Sk - mS).matrix().squaredNorm() / (nbProj - 1));
  }

  ArrayXd fftshift_2d(const double *X, size_t row, size_t col) 
  {
    // 2D fftshift of array X (row-major flattend). The dimension of X is (row, col) 
    // for even dimension only
    assert(row % 2 ==0);
    assert(col % 2 ==0);
    
    ArrayXd Y(row * col);
    size_t hrow = row/2;	// half row
    size_t hcol = col/2;	// half column

    for(size_t r =0; r<hrow; r++)
      for(size_t c=0; c<hcol; c++) {
	Y[c+r*col] = X[c+hcol+(r+hrow)*col];
	Y[c+hcol+r*col] = X[c+(r+hrow)*col];
	Y[c+(r+hrow)*col] = X[c+hcol+r*col];
	Y[c+hcol+(r+hrow)*col] = X[c+r*col];
	// Y[c+r*col] = X[c+col+(r+row)*col];
	// Y[c+col+r*col] = X[c+(r+row)*col];
	// Y[c+(r+row)*col] = X[c+col+r*col];
	// Y[c+col+(r+row)*col] = X[c+r*col];
      }
    return Y;
  }

  void fftshift_2d(const fftw_complex *X, size_t row, size_t col, fftw_complex *Y) 
  {
    // 2D fftshift of array X (row-major flattend). The dimension of X is (row, col) 
    // for even dimension only
    assert(row % 2 ==0);
    assert(col % 2 ==0);

    size_t hrow = row/2;	// half row
    size_t hcol = col/2;	// half column

    for(size_t r =0; r<hrow; r++)
      for(size_t c=0; c<hcol; c++) {
	Y[c+r*col][0] = X[c+hcol+(r+hrow)*col][0];
	Y[c+hcol+r*col][0] = X[c+(r+hrow)*col][0];
	Y[c+(r+hrow)*col][0] = X[c+hcol+r*col][0];
	Y[c+hcol+(r+hrow)*col][0] = X[c+r*col][0];

	Y[c+r*col][1] = X[c+hcol+(r+hrow)*col][1];
	Y[c+hcol+r*col][1] = X[c+(r+hrow)*col][1];
	Y[c+(r+hrow)*col][1] = X[c+hcol+r*col][1];
	Y[c+hcol+(r+hrow)*col][1] = X[c+r*col][1];
      }
  }

  ArrayXd zeropadding(const ArrayXd &X, const Array2i &dim, const Array2i &dim1)
  {		 
    // symmetric zero padding, from dimension dim to dim1
    // dim and dim1 must be both even or odd
    assert(dim1.y()>dim.y() && dim1.x()>dim.x());

    ArrayXd Y(dim1.prod());
    Y.setZero();

    size_t r0 = (size_t)((dim1.y() - dim.y())/2.);
    size_t c0 = (size_t)((dim1.x() - dim.x())/2.);
    size_t r1 = (size_t)((dim1.y() + dim.y())/2.);
    size_t c1 = (size_t)((dim1.x() + dim.x())/2.);

    for (size_t row=0; row<dim1.y(); row++) // row iteration
      for (size_t col=0; col<dim1.x(); col++) // col iteration
	if ((row>=r0 && row<r1) && (col>=c0 && col<c1))
	  Y[col+row*dim1.x()] = X[(col-c0) + (row-r0)*dim.x()];
    
    return Y;	  
  }

  inline double square_abscomplex(fftw_complex x) 
  {
    return x[0]*x[0]+x[1]*x[1];
  }

  inline double abscomplex(fftw_complex x) 
  {
    return sqrt(x[0]*x[0]+x[1]*x[1]);
  }

  void zoomin_interpl(const double *X0, const Array2i &vshape, double *Y0, size_t upSpl, int method) 
  {
    // Zoom in a 2d array X0 of dimension vshape by a factor upSpl using interpolation techniques
    // method : 0 for double of pixel, 1 for interlaced interpolation
    
  }

  void zoomin_fft(const double *X0, const Array2i &vshape, double *Y0, size_t upSpl, int method, double *W) 
  {
    // Zoom in a 2d array X0 of dimension vshape by a factor upSpl using FFT techniques
    // method : 0 for FC(Fourier Coeffs) periodization, 1 for 
    // and 2 for FC zeropadding.
    // W : 2d window function 

    Array2i ivshape = vshape * upSpl;

    size_t memsize_X = sizeof(fftw_complex) * vshape.prod();
    size_t memsize_Y = sizeof(fftw_complex) * ivshape.prod();

    fftw_complex *X, *Y;		// X0 and Y0 embedded in higher dimension and in complex form
    X = (fftw_complex *)fftw_malloc(memsize_X);
    Y = (fftw_complex *)fftw_malloc(memsize_Y);

    fftw_complex *FX, *FY;	// FFT of X and Y
    FX = (fftw_complex *)fftw_malloc(memsize_X);
    FY = (fftw_complex *)fftw_malloc(memsize_Y);

    // Plan for forward and inverse FFT
    fftw_plan planX, planY;

    // first dimension is row, then column
    planX = fftw_plan_dft_2d(vshape.y(), vshape.x(), X, FX, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(planX);		// forward transform

    // Filter 
    for(int u = 0; u<ivshape.y(); u++)
      for(int v = 0; v<ivshape.x(); v++) {
	// Following code is by periodization of FX
	FY[v + u*ivshape.x()][0] = FX[(v%vshape.x()) + (u%vshape.y())*vshape.x()][0]; // real part
	FY[v + u*ivshape.x()][1] = FX[(v%vshape.x()) + (u%vshape.y())*vshape.x()][1]; // imaginary part
      }

    planY = fftw_plan_dft_2d(ivshape.y(), ivshape.x(), FY, Y, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(planY);		// backward transform
  
    // for (size_t n = 0; n<inbNode; n++) {
    //   // Take only the active nodes value and only the real part
    //   // double nY = Y[iIdx[n]][0]*Y[iIdx[n]][0] + Y[iIdx[n]][1]*Y[iIdx[n]][1];
    //   // Y0[n] = sqrt(nY) / (double)(ivshape.prod()); // Don't forget the normalization!!
    //   Y0[n] = Y[iIdx[n]][0] / (double)(ivshape.prod()); // Don't forget the normalization!!
    // }

    fftw_destroy_plan(planX);
    fftw_destroy_plan(planY);
    fftw_free(FX); fftw_free(FY);
    fftw_free(X); fftw_free(Y);
  }

  void cudaSafe( cudaError_t error, const string &message)
  {
    if(error!=cudaSuccess) { 
      fprintf(stderr,"CUDA Error (%i) in %s: %s\n", error, message.c_str(), cudaGetErrorString(error)); 
      exit(-1);	
    }
  }

  void test_device(int idev) 
  {
    int deviceCount;
    //char *device = NULL;
    //unsigned int flags;
    cudaDeviceProp deviceProp;
  
    cudaSafe(cudaGetDeviceCount(&deviceCount), "");
    // idev = atoi(device);
    if(idev >= deviceCount || idev < 0){
      fprintf(stderr, "Invalid device number %d, using default device 0.\n",
	      idev);
      idev = 0;
    }

    cudaSafe(cudaSetDevice(idev), "");

    /* Verify the selected device supports mapped memory and set the device
       flags for mapping host memory. */

    cudaSafe(cudaGetDeviceProperties(&deviceProp, idev), "");

    if(!deviceProp.canMapHostMemory) {
      fprintf(stderr, "Device %d cannot map host memory!\n", idev);
      printf("PASSED");
      exit(-1);
    }
    else
      fprintf(stderr, "Device %d support host memory mapping!\n", idev);
  }

  void setActiveGPU(int dev) {
    cudaDeviceProp deviceProp;
    if (dev < 0) {
      deviceProp.major = 2;
      deviceProp.minor = 0;
      cudaSafe(cudaChooseDevice(&dev, &deviceProp), "Tools::setActiveGPU() : cudaChooseDevice()");    
    }
    cudaSafe(cudaSetDevice(dev), "Tools::setActiveGPU() : cudaSetDevice()");
    
    cudaSafe(cudaGetDevice(&dev), "Tools::setActiveGPU() : cudaGetDevice(&dev)");
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("GPU device in use : %d (%s)\n", dev, deviceProp.name);
  }
  
  string itoa_fixlen3(int n)
  {
    char tt[3];
    if (n<10)
      sprintf(tt,"00%d", n);
    else if (n<100)
      sprintf(tt,"0%d", n);
    else
      sprintf(tt,"%d", n);
    return string(tt);
  }

  double log_factorial(size_t N)
  {
    // Approximation of log factorial by
    // log n! ~ n(log n) - n + (log(n + 4 n^2 + 8 n^3))/6 + (log \pi)/2
    // which is a better approximation (exp(log n!)) than Stirling formula.
    double res = 0;

    if (N>100) 
      res = N*log(N*1.) - N + log(1.*(N + 4*N*N + 8*N*N*N))/6 + log(M_PI)/2;
    else if (N < 2)
      res = 0;
    else
      for (size_t n=2; n<=N; n++)
	res += log(n);
    return res;
  }

  template<class T>
  T multi_imsum(const vector <T> &X)
  {
    T Y = X[0];
    for (size_t n = 1; n<X.size(); n++)
      Y += X[n];
    return Y;
  }
  template ArrayXd multi_imsum<ArrayXd>(const vector <ArrayXd> &X);
  template ImageXXd multi_imsum<ImageXXd>(const vector <ImageXXd> &X);

  template<class T>
  vector<T> multi_imcumsum(const vector <T> &X)
  {
    vector<T> res;
    T Y = X[0];
    res.push_back(Y);
    for (size_t n = 1; n<X.size(); n++) {
      Y += X[n];
      res.push_back(Y);
    }
    return res;
  }

  template vector<ArrayXd> multi_imcumsum<ArrayXd>(const vector <ArrayXd> &X);
  template vector<ImageXXd> multi_imcumsum<ImageXXd>(const vector <ImageXXd> &X);

  // ImageXXd resampling_piximg(const double *X, int rowX, int colX, int rowY, int colY, double pixSizeY)
  // {

  //   double pixSizeX = 1.;
  //   ArrayXd Y = ArrayXd::Zero(rowY * colY);
  //   Array2d sizeObj(colX * pixSizeX, rowX * pixSizeX);
  //   Array2d pos; // = pix2pos(rr, cc, rowY, colY, pixSizeY);

  //   for (int rr = 0; rr<rowY; rr++)
  //     for (int cc = 0; cc<colY; cc++) {
  //   	pos.x() = (cc - colY/2) * pixSizeY + sizeObj.x()/2;
  //   	pos.y() = sizeObj.y()/2 - (rowY/2 - rr) * pixSizeY;
  //   	int col = (int)trunc(pos.x() / pixSizeX);
  //   	int row = (int)trunc(pos.y() / pixSizeX);
  //   	Y[cc + rr * colY] = (col<0 || col>=colX || row<0 || row>=rowX) ? 0 : X[col + row * colX];
  //     }
  //   return Map<ImageXXd>(Y.data(), rowY, colY);
  // }

  ImageXXd resampling_piximg(const double *X, int rowX, int colX, int rowY, int colY, double pixSizeY, bool oserr)
  {
    // Introduce a little bit of error to make the interpolation less favorable to pixel.
    // The blob image use hexagonal grid, and the interpolation often contains a small offset which makes
    // its SNR inferior to an equivalent pixel image. This is a default of SNR criteria.
    // To make a fair comparaison, we pertubate the pixel interpolation function by introducing a small offset error.
    ArrayXd Y = ArrayXd::Zero(rowY * colY);
    Array2d sizeObj(colX, rowX);
    Array2d pos;
    int col, row, idx;

    for (int rr = 0; rr<rowY; rr++)
      for (int cc = 0; cc<colY; cc++) {
    	pos.x() = (cc - colY/2) * pixSizeY + sizeObj.x()/2;
    	pos.y() = sizeObj.y()/2 - (rowY/2 - rr) * pixSizeY;
	col = (int)trunc(pos.x());
	row = (int)trunc(pos.y());
	idx = (oserr) ? (cc+1 + rr * colY) : (cc + rr * colY); // Offset error introduced	  
    	Y[idx % (rowY * colY)] = (col<0 || col>=colX || row<0 || row>=rowX) ? 0 : X[col + row * colX];
      }
    return Map<ImageXXd>(Y.data(), rowY, colY);
  }

  ImageXXd resampling_piximg(const double *X, const Array2i &dimObjX, const Array2i &dimObjY, double pixSizeY, bool oserr)
  {
    return resampling_piximg(X, dimObjX.y(), dimObjX.x(), dimObjY.y(), dimObjY.x(), pixSizeY, oserr);
  }

  ImageXXd resampling_piximg(const double *X, const Array2i &dimObjX, int dim, bool oserr)
  {
    // fit the longer side of original image X with dim
    int rowX = dimObjX.y();
    int colX = dimObjX.x();

    double cf = rowX*1./colX;
    int nrow, ncol;
    double pixSize;
    if (rowX>=colX) {
      nrow = dim;
      ncol = (int)ceil(nrow / cf);
      pixSize = rowX *1./ nrow;
    }
    else {
      ncol = dim;
      nrow = (int)ceil(ncol * cf);
      pixSize = colX *1./ ncol;
    }
    return resampling_piximg(X, rowX, colX, nrow, ncol, pixSize, oserr);
  }

  ImageXXd resampling_piximg(const double *X, int rowX, int colX, int nrow, int ncol, bool oserr)
  {
    // fit the longer side of new image with that of original image X
    double pixSize;
    if (nrow>=ncol) {
      pixSize = rowX *1./ nrow;
    }
    else {
      pixSize = colX *1./ ncol;
    }
    return resampling_piximg(X, rowX, colX, nrow, ncol, pixSize, oserr);
  }

  ArrayXi random_permutation(size_t N, size_t M) {
    const gsl_rng_type * T;
    gsl_rng * r;
     
    gsl_permutation * p = gsl_permutation_alloc (N);
    gsl_permutation * q = gsl_permutation_alloc (N);
     
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
     
    //printf ("initial permutation:");  
    gsl_permutation_init (p);
    //gsl_permutation_fprintf (stdout, p, " %u");
    //printf ("\n");
    
    //printf (" random permutation:");  
    gsl_ran_shuffle (r, p->data, N, sizeof(size_t));
    //gsl_permutation_fprintf (stdout, p, " %u");
    //printf ("\n");
    ArrayXi Perm(M);
    for (size_t m=0; m<M; m++)
      Perm[m] = p->data[m];

    // printf ("inverse permutation:");  
    // gsl_permutation_inverse (q, p);
    // gsl_permutation_fprintf (stdout, q, " %u");
    // printf ("\n");
    
    gsl_permutation_free (p);
    //gsl_permutation_free (q);
    gsl_rng_free (r);
    
    return Perm;
  }

  ArrayXd joint(const vector<ArrayXd> &Y)
  {
    int N=0;
    for (int n=0; n<Y.size(); n++)
      N += Y[n].size();
    ArrayXd X(N);
    size_t offset = 0;
  
    for (int n =0; n<Y.size(); n++) {
      X.segment(offset, Y[n].size()) = Map<ArrayXd> (Y[n].data(), Y[n].size());
      offset += Y[n].size();
    }
    return X;
  }

  vector<ArrayXd> separate(const ArrayXd &X, vector<ArrayXd> &Y) 
  {
    vector<ArrayXd> Q; 
    Q.resize(Y.size());
    size_t offset = 0;
    for (int n=0; n<Y.size(); n++) {
      Q[n] = X.segment(offset, Y[n].size());
      offset += Y[n].size();
    }
    return Q;
  }

  template <class T>
  //T Nth_maxCoeff_rand(const T *X, size_t M, size_t N, const gsl_rng *rng=0) 
  T Nth_maxCoeff(const T *X, size_t M, size_t N) 
  {
    // Find the N-th maximum element of an array X with size M
    
    //let r be chosen uniformly at random in the range 1 to length(A)    
    size_t K = (size_t)ceil(N/2.);

    // if (rng == NULL)
    //   K = (size_t)ceil(N/2.);
    // else
    //   K = gsl_rng_uniform_int(rng, N);

    T pivot = X[K];    //let pivot = A[r]
    T res;	       // result to return

    // split into a pile X1 of big elements and X2 of small elements
    T *X1, *X2;
    X1 = (T*)malloc(M*sizeof(T));
    X2 = (T*)malloc(M*sizeof(T));
    size_t S1=0, S2=0;		// size of X1 and X2

    for(size_t n=0; n<M; n++) {
      if(X[n] > pivot)
	X1[S1++] = X[n]; // append A[i] to A1
      else if (X[n] < pivot)
	X2[S2++] = X[n]; // append A[i] to A2
    }

    if (S1 > N) {      // it's in the pile of big elements
      res = Nth_maxCoeff(X1, S1, N);
    }
    else if (S2 > M - N) {      // it's in the pile of small elements
      res = Nth_maxCoeff(X2, S2, S2-(M-N));
    }
    else
      res = pivot;

    // it's equal to the pivot
    free(X1); 
    free(X2);
    return res;
  }

  template double Nth_maxCoeff<double> (const double *X, size_t M, size_t N);
  template int Nth_maxCoeff<int> (const int *X, size_t M, size_t N);

  // template <class T>
  // T Nth_maxCoeff(const T *X, size_t M, size_t N) 
  // {
  //   // Find the N-th maximum element of an array X with size M
    
  //   //let r be chosen uniformly at random in the range 1 to length(A)    
  //   size_t K = (size_t) N/2.;
  //   T pivot = X[K];    //let pivot = A[r]

  //   // split into a pile X1 of big elements and X2 of small elements
  //   T *X1, *X2;
  //   X1 = (T*)malloc(M*sizeof(T));
  //   X2 = (T*)malloc(M*sizeof(T));
  //   size_t S1=0, S2=0;		// size of X1 and X2

  //   for(size_t n=0; n<M; n++) {
  //     if(X[n] > pivot)
  // 	X1[S1++] = X[n]; // append A[i] to A1
  //     else if (X[n] < pivot)
  // 	X2[S2++] = X[n]; // append A[i] to A2
  //   }

  //   if (S1 > N) {      // it's in the pile of big elements
  //     return Nth_maxCoeff(X1, S1, N);
  //   }
  //   else if (S2 > M - N) {      // it's in the pile of small elements
  //     return Nth_maxCoeff(X2, S2, S2-(M-N));
  //   }

  //   // it's equal to the pivot
  //   free(X1); 
  //   free(X2);
  //   return pivot;
  // }

  // template double Nth_maxCoeff<double> (const double *X, size_t M, size_t N);
  // template int Nth_maxCoeff<int> (const int *X, size_t M, size_t N);

  int l0norm(const ArrayXd &X)
  {
    ArrayXi S = (X.abs()>0).select(1, ArrayXi::Zero(X.size()));
    return S.sum();
  }
}

  // /* Rotation of a vector by theta */
  // Array2d rotation(const Array2d &A, double theta)
  // /* Anti-clockwise rotation of A by theta */
  // {
  //   return Array2d(cos(theta)*A.x() - sin(theta)*A.y(),
  // 		   sin(theta)*A.x() + cos(theta)*A.y());
  // }

  // double SNR(const ArrayXXd &X, const ArrayXXd &Y)
  // {
  //   // Calculate the Signal to Noise Ratio(SNR) of
  //   // noisy signal X compared to noiseless Y.
  //   // X and Y are normalized.
  //   // SNR(X,Y) := 10 * log10(e_s / e_n)    
  //   // with e_s and e_n the enery of signal and noise
  //   assert(X.rows() == Y.rows());
  //   assert(X.cols() == Y.cols());

  //   ArrayXd X0 = Map<ArrayXd> (X.data(), X.size());
  //   ArrayXd Y0 = Map<ArrayXd> (Y.data(), Y.size());

  //   // X0 = X0 - X0.mean();
  //   // Y0 = Y0 - Y0.mean();
  //   X0 /= X0.matrix().norm();
  //   Y0 /= Y0.matrix().norm();

  //   // Energy of noise 
  //   ArrayXd Noise = X0 - Y0;
  //   double mN = Noise.mean();
  //   double eN = (Noise - mN).matrix().squaredNorm();

  //   // Energy of signal
  //   double mS = Y0.mean();
  //   double eS = (Y0 - mS).matrix().squaredNorm();

  //   return 10 * log10(eS / eN);
    
  //   // ArrayXd X0 = Map<ArrayXd> (X.data(), X.size());
  //   // ArrayXd Y0 = Map<ArrayXd> (Y.data(), Y.size());
  //   // X0 = X0 - X0.mean();
  //   // Y0 = Y0 - Y0.mean();
  //   // X0 /= X0.matrix().norm();
  //   // Y0 /= Y0.matrix().norm();
  //   // // X0 /= X0.maxCoeff();
  //   // // Y0 /= Y0.maxCoeff();
  //   // return -20 * log10((X0-Y0).matrix().norm());
  // }

