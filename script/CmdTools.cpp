#define cimg_medcon_path "~/lami3/bin"

#include "CmdTools.hpp"

ostream& operator<<(ostream& out, const SimConfig& conf) {
  out<<*(AcqConfig *)&conf<<endl;
  out<<"----Sinogram simulation settings----"<<endl;
  out<<"Size of object (H X W) : "<<conf.sizeObj.y()<<" X "<<conf.sizeObj.x()<<endl;
  out<<"Dimension of phantom (R X C) : "<<conf.dimObj.y()<<" X "<<conf.dimObj.x()<<endl;
  out<<"Phantom pixel size : "<<conf.spObj<<endl;
  out<<"Projection method : "<<conf.method<<endl;
  out<<"Phantom file : "<<conf.phantom<<endl;
  out<<"Source intensity : "<<conf.phs<<endl;
  out<<"Sinogram noise energy : "<<conf.noise_energy<<endl;
  out<<"Sinogram SNR : "<<conf.snr<<endl;
  return out;
}

namespace CmdTools {
  string creat_outpath(const string &outpath)
  {
    int n=1;    
    char buffer[256];
    sprintf(buffer, "%s", outpath.c_str());

    while (mkdir(buffer, 0777) == -1 && n<256) {
      if (errno == EEXIST) {
	cout<<"File exists : "<<buffer<<endl;
	sprintf(buffer, "%s_%d", outpath.c_str(), n++);
      }
      else exit(0);
    }
    return string(buffer);
  }

  string rename_outpath(const string &oldname, const string &newname)
  {
    int n=1;    
    char  buffer_old[256], buffer_new[256];
    sprintf(buffer_old, "%s", oldname.c_str());
    sprintf(buffer_new, "%s", newname.c_str());

    while (rename(buffer_old, buffer_new) == -1 && n<256) {
      if (errno == EEXIST) {
	cout<<"File exists : "<<buffer_new<<endl;
	sprintf(buffer_new, "%s_%d", newname.c_str(), n++);
      }
      else exit(0);
    }
    return string(buffer_new);
  }

  // void loadarray(const char *fname, ArrayXd &Y)
  // {
  //   // Load a binary data file into a double array
  //   // Inputs :
  //   // fname : file name
  //   // Y : a preallocated Eigen array, its size must
  //   // match the number of elements in file fname

  //   ifstream in(fname, ios::in | ios::binary);
  //   if (!in) {
  //     cout <<"Cannot open file : "<<fname<<endl;
  //     exit(1);
  //   }
  //   in.read((char *)Y.data(), sizeof(double)*Y.size());
  //   in.close();
  // }

  void save_acqcarm(const SimConfig &conf, const string &fname)
  {
    // Save a CARM-acquisition configuration to file
    ofstream fout; //output configuration file
    fout.open(fname.data(), ios::out);
    if (!fout) {
      cout <<"Cannot open file : "<<fname<<endl;
      exit(1);
    }

    fout<<"[CARM acquisition configuration]"<<endl;
    fout<<"fanbeam="<<conf.fanbeam<<endl;
    fout<<"nbProj="<<conf.nbProj_total<<endl;
    fout<<"pSrc=";
    for (int n = 0; n<conf.nbProj_total; n++)
      fout<<conf.pSrc[n]<<", ";
    // if (n<nbProj-1) 
    //   fout<<pSrc[n]<<", ";
    // else 
    //   fout<<pSrc[n];
    fout<<endl;
    fout<<"rSrc="<<conf.rSrc[0]<<endl;
    fout<<"rDet="<<conf.rDet[0]<<endl;
    //    fout<<"rtDet="<<conf.rtDet[0]<<endl;
    fout<<"sizeDet="<<conf.sizeDet<<endl;
    fout<<"pixDet="<<conf.pixDet<<endl;
    fout<<"spDet="<<conf.spDet<<endl;
    fout<<"diamFOV="<<conf.diamFOV<<endl;
    fout<<"sizeObj="<<conf.sizeObj[0]<<", "<<conf.sizeObj[1]<<endl;
    //fout<<"diamROI="<<conf.diamROI<<endl;
    fout<<"dimObj="<<conf.dimObj[0]<<", "<<conf.dimObj[1]<<endl;
    fout<<"spObj="<<conf.spObj<<endl;
    fout<<"method="<<conf.method<<endl;
    fout<<"phantom="<<conf.phantom<<endl;
    fout<<"phs="<<conf.phs<<endl;
    fout<<"noise_energy="<<conf.noise_energy<<endl;
    fout<<"noise_std="<<conf.noise_std<<endl;
    fout<<"snr="<<conf.snr<<endl;

    fout.close();
  }

  SimConfig load_acqcarm(const string &fname, double box, double roi)
  {
    // Load a CARM-acquisition configuration from file, only non-optional properties are loaded

    bool fanbeam = true;
    int nbProj = 0;
    // ArrayXd _pSrc;
    vector<double> vpSrc;
    Array2d sizeObj(0,0);
    Array2i dimObj(0,0);
    double spObj = 0;
    int pixDet = 0;
    double rDet = 0;
    //    double rtDet = 0;
    double rSrc = 0;
    double sizeDet = 0;  
    double diamFOV = 0;
    size_t phs = 0;
    double noise_energy = 0;
    double noise_std = 0;
    double snr = 0;
    double beamDiv;
    string method = "";
    string phantom = "";
    //double blob_dilate = 0;

    ifstream fin; //output configuration file
    fin.open(fname.data(), ios::in);
    if (!fin) {
      cout <<"Cannot open file : "<<fname<<endl;
      exit(1);
    }
    string line, fieldname, fieldvalue;
    getline(fin, line);		// read off the title

    while(getline(fin, line)) {  
      stringstream recStream(line);
      // extract first the property name  
      getline(recStream, fieldname, '=');

      removeSpaces(fieldname);  

      if (fieldname == "fanbeam") {
	getline(recStream, fieldvalue);
	fanbeam = atoi(fieldvalue.data());
      }
      else if (fieldname == "nbProj") {
	getline(recStream, fieldvalue);
	nbProj = atoi(fieldvalue.data());
      }
      else if (fieldname == "pSrc") {
	while (getline(recStream, fieldvalue, ',')) {
	  double p = atof(fieldvalue.data());	  
	  vpSrc.push_back(p);
	  //_pSrc<<atof(fieldvalue.data());
	  //cout<<p<<" ";
	}
      }
      else if (fieldname == "rSrc") {
	getline(recStream, fieldvalue);
	rSrc = atof(fieldvalue.data());
      }
      else if (fieldname == "rDet") {
	getline(recStream, fieldvalue);
	rDet = atof(fieldvalue.data());
      }
      // else if (fieldname == "rtDet") {
      // 	getline(recStream, fieldvalue);
      // 	rtDet = atof(fieldvalue.data());
      // }
      else if (fieldname == "sizeDet") {
	getline(recStream, fieldvalue);
	sizeDet = atof(fieldvalue.data());
      }
      else if (fieldname == "pixDet") {
	getline(recStream, fieldvalue);
	pixDet = atoi(fieldvalue.data());
      }
      // else if (fieldname == "sizeObj") {
      // 	//getline(recStream, fieldvalue);
      // 	getline(recStream, fieldvalue, ',');
      // 	sizeObj.x() = atof(fieldvalue.data());
      // 	getline(recStream, fieldvalue);
      // 	sizeObj.y() = atof(fieldvalue.data());
      // }
      // else if (fieldname == "dimObj") {
      // 	//getline(recStream, fieldvalue);
      // 	getline(recStream, fieldvalue, ',');
      // 	dimObj.x() = atoi(fieldvalue.data());
      // 	getline(recStream, fieldvalue);
      // 	dimObj.y() = atoi(fieldvalue.data());
      // }
      // else if (fieldname == "spObj") {
      // 	getline(recStream, fieldvalue);
      // 	spObj = atof(fieldvalue.data());
      // }
      else if (fieldname == "phantom") {
      	getline(recStream, fieldvalue);
      	phantom = fieldvalue;
      }
      // else if (fieldname == "method") {
      // 	getline(recStream, fieldvalue);
      // 	method = fieldvalue;
      // }
      // else if (fieldname == "phs") {
      // 	getline(recStream, fieldvalue);
      // 	phs = atoi(fieldvalue.data());
      // }    
      // else if (fieldname == "noise_energy") {
      // 	getline(recStream, fieldvalue);
      // 	noise_energy = atof(fieldvalue.data());
      // }
      // else if (fieldname == "noise_std") {
      // 	getline(recStream, fieldvalue);
      // 	noise_std = atof(fieldvalue.data());
      // }
      // else if (fieldname == "snr") {
      // 	getline(recStream, fieldvalue);
      // 	snr = atof(fieldvalue.data());
      // }
    }
    
    diamFOV = Acq::eval_diamFOV(fanbeam, rSrc, rDet, sizeDet);
    double diamROI;

    if (phantom != "") { // If pixel phantom image file is given in the acq config file, 
      diamROI = diamFOV; 
      ImageXXd im = CmdTools::imread(phantom);
      if (im.rows() > 0 and im.cols() > 0) { // and if it's valid,
	dimObj.x() = im.cols();
	dimObj.y() = im.rows();
	double ldiag = sqrt(dimObj.x()*dimObj.x() + dimObj.y()*dimObj.y());
	// By convention, (the same used for generate the sinogram
	// from phantom), sizeObj is the biggest rectangular
	// containned in the FOV
	sizeObj.x() = diamFOV * dimObj.x() * 1./ ldiag; 
	sizeObj.y() = diamFOV * dimObj.y() * 1./ ldiag;
      }
      else {
	cerr<<"Cannot locate the phantom pixel image : "<<phantom<<endl;
	exit(0);
      }
    }
    else { //if (sizeObj.x()<=0 || sizeObj.y()<=0) { 
      // no reference image is given,  determine the reconstruction/interpolation region size
      diamROI = diamFOV * roi; // restriction of effective reconstruction region
      sizeObj.x() = fmin(diamFOV * box, diamROI);
      sizeObj.y() = fmin(diamFOV * box, diamROI);
      // if(box) { // sizeObj is the square included in FOV if box is true
      // 	sizeObj.x() = diamROI / sqrt(2);
      // 	sizeObj.y() = diamROI / sqrt(2);
      // }      
      // else {	// sizeObj is the square including FOV if box is false
      // 	sizeObj.x() = diamROI;
      // 	sizeObj.y() = diamROI;
      // }
      dimObj.x() = 512;	
      dimObj.y() = 512;
    }
    // cout<<"CmdTools::load_acqcarm() diamFOV = "<<diamFOV<<endl;
    // cout<<"CmdTools::load_acqcarm() roi = "<<roi<<endl;

    // Normalize the whole system such that the new diamFOV = STD_DIAMFOV
    double FC = STD_DIAMFOV / diamFOV;

    ArrayXd _rSrc(nbProj);
    _rSrc.setConstant(rSrc);

    ArrayXd _rDet(nbProj);
    _rDet.setConstant(rDet);

    ArrayXd _pSrc(nbProj);
    for (int n=0; n<nbProj; n++)
      _pSrc[n] = vpSrc[n];
    // ArrayXd vrtDet(nbProj);
    // vrtDet.setConstant(rtDet);
    
    double spDet = sizeDet / pixDet;
  
    // Create AcqConfig object
    SimConfig conf(fanbeam,
		   nbProj,
		   _rSrc * FC,
		   _pSrc,
		   _rDet * FC,
		   pixDet,
		   sizeDet * FC,
		   diamFOV * FC,
		   "CARM",
		   sizeObj * FC,
		   diamROI * FC,
		   dimObj,
		   spObj * FC,
		   method,
		   phantom,
		   phs,
		   noise_energy,
		   noise_std,
		   snr);
    return conf;
  }

  SimConfig extract_config(const SimConfig &conf, int nbProj, bool endpoint) 
  {
    ArrayXd rSrc(nbProj);
    ArrayXd pSrc(nbProj);
    ArrayXd rDet(nbProj);

    ArrayXi idx = Tools::subseq(conf.nbProj_total, nbProj, endpoint);

    for (int p=0; p<nbProj; p++) {
      rSrc[p] = conf.rSrc[idx[p]];
      pSrc[p] = conf.pSrc[idx[p]];
      rDet[p] = conf.rDet[idx[p]];
    }
    
    return SimConfig(conf.fanbeam,
		     nbProj,
		     rSrc,
		     pSrc,
		     rDet,
		     conf.pixDet,
		     conf.sizeDet,
		     conf.diamFOV,
		     conf.acqname,
		     conf.sizeObj,
		     conf.diamROI,
		     conf.dimObj,
		     conf.spObj,
		     conf.method,
		     conf.phantom,
		     conf.phs,
		     conf.noise_energy,
		     conf.noise_std,
		     conf.snr);
  }

  // SimConfig nrml_config(const SimConfig &conf, double diamFOV)
  // {
  //   double FC = diamFOV / conf.diamFOV;
    
  //   return SimConfig (conf.fanbeam,
  // 		      conf.nbProj_total,
  // 		      conf.rSrc * FC,
  // 		      conf.pSrc,
  // 		      conf.rDet * FC,
  // 		      conf.pixDet,
  // 		      conf.sizeDet * FC,
  // 		      diamFOV,
  // 		      conf.acqname,
  // 		      conf.sizeObj * FC,
  // 		      conf.diamROI * FC,
  // 		      conf.dimObj,
  // 		      conf.spObj * FC,
  // 		      conf.method,
  // 		      conf.phantom,
  // 		      conf.phs,
  // 		      conf.noise_energy,
  // 		      conf.noise_std,
  // 		      conf.snr);
  //   //return conf;
  // }

  ArrayXd loadarray(const string &fname, size_t N)
  {
    // Load a binary data file into a double array
    // Inputs :
    // fname : file name
    // N : number of elements to be loaded
    // Outputs : 1D Eigen double array
    
    ArrayXd Y(N);
    ifstream in(fname.data(), ios::in | ios::binary);
    if (!in) {
      cout <<"Cannot open file : "<<fname<<endl;
      exit(1);
    }
    in.read((char *)Y.data(), sizeof(double)*N);
    in.close();
    return Y;
  }


  ArrayXd loadarray_float(const string &fname, size_t N)
  {
    // Load a binary data file into a double array
    // Inputs :
    // fname : file name
    // N : number of elements to be loaded
    // Outputs : 1D Eigen double array

    ArrayXf Y(N);
    ifstream in(fname.data(), ios::in | ios::binary);
    if (!in) {
      cout <<"Cannot open file : "<<fname<<endl;
      exit(1);
    }
    in.read((char *)Y.data(), sizeof(float)*N);
    in.close();
    ArrayXd D(N);
    for (size_t n=0; n<N; n++)
      D[n] = Y[n];
    return D;
  }

  ArrayXd loadarray(const string &fname, size_t N, bool single_precision, bool endian_swap)
  {
    // Load a binary data file into a double array
    // Inputs :
    // fname : file name
    // N : number of elements to be loaded
    // nbyte : size of data type in number of bytes. 8 for double, 4 for float, 2 for short int.
    // Outputs : 1D Eigen double array

    char *Y;
    int nbyte = single_precision ? 4 : 8;
    Y = new char[N * nbyte];
    //ArrayXf Y(N);
    ifstream in(fname.data(), ios::in | ios::binary);
    if (!in) {
      cout <<"Cannot open file : "<<fname<<endl;
      exit(1);
    }
    in.read(Y, nbyte * N);
    in.close();

    ArrayXd D(N);
    if (single_precision){
      Map<ArrayXf> toto((float *)Y, N);
      //Map<ArrayXf> toto(Y, N);
      for (size_t n=0; n<N; n++) {
	if (endian_swap)
	  D[n] = (double)endswap(toto[n]);
	else 
	  D[n] = (double) toto[n];
      }
    }
    else {
      // double tmp; char *a = (char *)&tmp;
      // for (size_t n=0; n<N; n++) {
      // 	if (lendian) {
      // 	  for(int k=0; k<nbyte; k++)
      // 	    a[k] = Y[n*nbyte + nbyte - 1 - k];
      // 	}
      // 	else {
      // 	  for(int k=0; k<nbyte; k++)
      // 	    a[k] = Y[n*nbyte + k];
      // 	}
      // 	D[n] = tmp;
      // }
      Map<ArrayXd> toto((double *)Y, N);
      for (size_t n=0; n<N; n++) {
	if (endian_swap)
	  D[n] = endswap(toto[n]);
	else
	  D[n] = toto[n];
      }
    }    
    delete [] Y;
    return D;
  }

  template<class T>
  void savearray(const T &X, const string &fname)
  {
    // save a double array to a binary data file
    // Inputs :
    // fname : file name
    // X : array to be saved
    char buffer[256];
    sprintf(buffer, "%s.dat", fname.c_str());

    ofstream out(buffer, ios::out | ios::binary);
    if (!out) {
      cout <<"Cannot open file : "<<buffer<<endl;
      exit(1);
    }
    out.write((char *)X.data(), sizeof(double)*X.size());
    out.close();
  }

  template void savearray<ArrayXd>(const ArrayXd &X, const string &fname);
  template void savearray<ImageXXd>(const ImageXXd &X, const string &fname);

  template<class T>
  void multi_savearray(const vector<T> &X, const string &fname)
  {
    char buffer[256];
    for (int n = 0; n<X.size(); n++) {      
      sprintf(buffer, "%s_scale_%d", fname.c_str(), n);
      savearray(X[n], buffer);
    }
  }

  template void multi_savearray<ArrayXd>(const vector<ArrayXd> &X, const string &fname);
  template void multi_savearray<ImageXXd>(const vector<ImageXXd> &X, const string &fname);

  // template <class T> void endswap(T *objp)
  // {
  //   //unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  //   char *memp = reinterpret_cast<char*>(objp);
  //   std::reverse(memp, memp + sizeof(T));
  // }

  template <class T> T endswap(T d)
  {
    T a;
    unsigned char *dst = (unsigned char *)&a;
    unsigned char *src = (unsigned char *)&d;

    for (int n=0; n<sizeof(T); n++)
      dst[n] = src[sizeof(T) - n - 1];
      
   return a;
  }
  template double endswap<double>(double d);
  template float endswap<float>(float d);

  //template<class T>
  ImageXXd imread(const string &fname)
  {
    CImg<double> im(fname.data());
    // cout<<"height="<<im.height()<<endl;
    // cout<<"width="<<im.width()<<endl;
    // double *data = new double[im.height() * im.width()];

    // for (int n = 0 ; n < im.height() * im.width(); n++)
    //   data[n] = (double) *(im.data() + n);

    Map<ImageXXd> X(im.data(), im.height(), im.width());
    // cout<<X.minCoeff()<<endl;
    // cout<<X.maxCoeff()<<endl;
    return (X - X.minCoeff()) / (X.maxCoeff() - X.minCoeff());    
  }

  // template ImageXXd imread<float>(const string &fname);
  // template ImageXXd imread<double>(const string &fname);

  void imshow(const ImageXXd &X, const string &title)
  {
    CImg<double> im(X.data(), X.cols(), X.rows()); //width (col) first, then height (row)
    im.display(title.data());
  }

  void imshow(const ArrayXd &X, int row, int col, const string &title)
  {
    CImg<double> im(X.data(), col, row);
    im.display(title.data());
  }

  void imshow(const double *X, int row, int col, const string &title)
  {
    CImg<double> im(X, col, row);
    im.display(title.data());
  }

  void multi_imshow(const vector<ArrayXd> &X, int row, int col, const string& title)
  {
    // ArrayXXd im = Tools::multi_imsum(X); // Show the sum of all scales images
    // imshow(im, title);

    char msg[256];
    for (int n = 0; n<X.size(); n++) {
      sprintf(msg, "%s scale %d", title.c_str(), n);
      imshow(X[n], row, col, msg);
    }
  }

  void multi_imshow(const vector<ImageXXd> &X, const string& title)
  {
    // ArrayXXd im = Tools::multi_imsum(X); // Show the sum of all scales images
    // imshow(im, title);

    char msg[256];
    for (int n = 0; n<X.size(); n++) {
      sprintf(msg, "%s scale %d", title.c_str(), n);
      imshow(X[n], msg);    
    }
  }

  void imsave(const double *X, int row, int col, const string &fname)
  {
    char buffer[256];
    sprintf(buffer, "%s.png", fname.c_str());
    Map<ArrayXd> Y(X, row*col);

    ArrayXd Z = Y;
    Z = (Y - Y.minCoeff()) / (Y.maxCoeff() - Y.minCoeff()) * 255;
    CImg<double> im(Z.data(), col, row);

    im.save(buffer);
  }

  void imsave(const ArrayXd &X, int row, int col, const string &fname)
  {
    imsave(X.data(), row, col, fname);
  }

  void imsave(const ImageXXd &X, const string &fname)
  {
    imsave(X.data(), X.rows(), X.cols(), fname);
  }

  void multi_imsave(const vector<ImageXXd> &X, const string &fname)
  {
    char buffer[256];
    for (int n = 0; n<X.size(); n++) {      
      sprintf(buffer, "%s_scale_%d", fname.c_str(), n);      
      imsave(X[n], buffer);
    }
  }

  void BenchMark_Op(LinOp &A, int N) {
    if (N<=0) return;
    cout<<A<<endl;
    ArrayXd x0 = ArrayXd::Ones(A.get_dimX());
    ArrayXd y0(A.get_dimY());

    // cout<<x0.mean()<<endl;
    // cout<<x0.size()<<" "<<A.get_dimX()<<endl;
    clock_t t0 = clock();
    for (int n=0; n<N; n++)
      A.forward(x0, y0);
    double t1 = (clock()-t0)/(double)CLOCKS_PER_SEC/N;
    printf("Forward operator takes in average %lf seconds\n", t1); 

    y0 = ArrayXd::Ones(A.get_dimY());
    t0 = clock();
    for (int n=0; n<N; n++)
      A.backward(y0, x0);
    t1 = (clock()-t0)/(double)CLOCKS_PER_SEC/N;
    printf("Backward projections takes in average %lf seconds\n\n", t1); 
  }

  void removeSpaces(string &stringIn )
  {
    string::size_type pos = 0;
    bool spacesLeft = true;
    
    while( spacesLeft )
      {
	pos = stringIn.find(" ");
	if( pos != string::npos )
	  stringIn.erase( pos, 1 );
	else
	  spacesLeft = false;
      }
    //return stringIn;
  } 

  string extractFilename(string &stringIn)
  {
    string::size_type pos_end = 0;
    string::size_type pos_start = 0;
    
    pos_end = stringIn.rfind(".");
    pos_start = stringIn.rfind("/");
    // cout<<pos_start<<endl;
    // cout<<pos_end<<endl;
    string::size_type len, ss;
    if (pos_start==string::npos) {
      ss = 0;
      len = (pos_end == string::npos) ? stringIn.size() : pos_end;
    }
    else {
      ss = pos_start+1;
      len = (pos_end == string::npos) ? stringIn.size()-pos_start-1: pos_end - pos_start-1;
    }
    return stringIn.substr(ss, len);
  } 
  
}


