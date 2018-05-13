// Various pieces of codes not used anymore

// 1. Previous version of save_algo_parms
// void save_algo_parms(const string &fname, const AlgoParms &AP, const BlobImage &BI, const ConvMsg *msg)
// {
//   // Save current algorithm's parameters in a file
//   ofstream fout; //output configuration file
//   fout.open(fname.data(), ios::out);
//   if (!fout) {
//     cout <<"Cannot open file : "<<fout<<endl;
//     exit(1);
//   }

//   fout<<"[BlobImage Parameters]"<<endl;
//   fout<<"fspls="<<AP.fspls<<endl;
//   fout<<"cut_off_err="<<AP.cut_off_err<<endl;
//   fout<<"fcut_off_err="<<AP.fcut_off_err<<endl;
//   fout<<BI<<endl;

//   fout<<"[Projector Parameters]"<<endl;
//   fout<<"stripint="<<AP.stripint<<endl;
//   fout<<"norm_projector="<<AP.norm_projector<<endl;
//   fout<<endl;

//   fout<<"[Algorithm Parameters]"<<endl;
//   fout<<"algo="<<AP.algo<<endl;
//   fout<<"besov="<<AP.besov<<endl;
//   if(AP.algo=="padm") {
//     fout<<"beta_rel="<<AP.beta_rel<<endl;
//     fout<<"beta="<<AP.beta<<endl;
//     fout<<"epsilon="<<AP.epsilon<<endl;
//     fout<<"gamma="<<AP.gamma<<endl;
//     fout<<"tau="<<AP.tau<<endl;
//   }
//   fout<<"tol="<<AP.tol<<endl;
//   fout<<"maxIter="<<AP.maxIter<<endl;

//   fout<<"rwIter="<<AP.rwIter<<endl;
//   fout<<"rwEpsilon="<<AP.rwEpsilon<<endl;
//   fout<<endl;

//   fout<<"[Results]"<<endl;
//   fout<<"nnz="<<AP.nnz<<endl;
//   fout<<"sparsity="<<AP.sparsity<<endl;

//   fout<<"snr="<<AP.snr<<endl;
//   fout<<"corr="<<AP.corr<<endl;
//   fout<<"uqi="<<AP.uqi<<endl;
//   fout<<"mse="<<AP.mse<<endl;
//   fout<<"si="<<AP.si<<endl;
//   fout<<"time="<<AP.time<<endl;

//   for (int n=0; n<AP.rwIter; n++) {
//     fout<<endl<<"Reweighted iteration : "<<n<<endl;
//     fout<<"niter="<<msg[n].niter<<endl;
//     fout<<"residual |Ax-b|="<<msg[n].res<<endl;
//     fout<<"l1 norm="<<msg[n].norm<<endl;
//   }

//   fout.close();
// }


// 2. About the mask
// ArrayXd Mask = 1 - BI->prod_mask(Xr, nz); // Attention : The mask is inverse, 0 for the presence of blob
// // vector<ArrayXd> V = BI->separate(Mask);
// // for (int m=0; m<V.size(); m++) {
// //  	//printf("Scale %d, min = %e, max = %e\n", m, V[m].minCoeff(), V[m].maxCoeff());
// // 	CmdTools::imshow(V[m], BI->bgrid[m]->vshape.y(), BI->bgrid[m]->vshape.x(), "Mask");	
// // }

// double xmax = Xr.abs().maxCoeff();
// for (size_t m=0; m<Xr.size(); m++) {
// 	toto[m] = 1/(fabs(Xr[m]) + epsilon * xmax);
// }
//  W = toto * Mask / toto.maxCoeff();
// cout<<"Wmax = "<<W.maxCoeff()<<endl;
// cout<<"Wmin = "<<W.minCoeff()<<endl;

//Xr *= (1-Mask);

// 3. About scale product
// vector<ArrayXd> Q = BI->interscale_product(Xr);
// if (nterm_prct > 0 and nterm_prct < 1)
// 	nterm = min((int)ceil(nterm_prct * P->get_dimX()), nonzero);

// for (int m=0; m<Q.size(); m++) {
//  	printf("Scale %d, min = %e, max = %e\n", m, Q[m].minCoeff(), Q[m].maxCoeff());
// 	CmdTools::imshow(Q[m], BI->bgrid[m+2]->vshape.y(), BI->bgrid[m+2]->vshape.x(), "weight");	
// }

// W = BI->ProdThresh(Xr, epsilon, n);

// offset = 0;
// for (int j=0; j<BI->get_nbScale(); j++) {
// 	int N = BI->bgrid[j]->nbNode; // Number of coefficients (nodes) of scale j
// 	double xmax = Xr.segment(offset, N).abs().maxCoeff(); // Maximum magnitude of coefficients of scale j
// 	toto.setZero(N);
// 	for (size_t m=0; m<N; m++) {
// 	  toto[m] = 1/(fabs(X[offset + m]) + epsilon * xmax);
// 	}
// 	W.segment(offset, N) = toto / toto.maxCoeff() * pow(BI->get_scaling(), 0.1*j);
// 	//W.segment(offset, N) = toto * pow(BI->get_scaling(), 0.1*j);
// 	offset += N;
// }

// vector<ArrayXd> V = BI->separate(Xr);

// for (int m=0; m<V.size(); m++) {
//  	printf("Scale %d, min = %e, max = %e\n", m, V[m].minCoeff(), V[m].maxCoeff());
// 	CmdTools::imshow(V[m], BI->bgrid[m]->vshape.y(), BI->bgrid[m]->vshape.x(), "weight");	
// }
      
// double xmax = Xr.abs().maxCoeff(); // Maximum magnitude of coefficients of scale j
// for (size_t m=0; m<Xr.size(); m++)
// 	W[m] = 1/(fabs(Xr[m]) + epsilon * xmax);
// W = W / W.maxCoeff();


// 4. About estimation of projector's norm
// ArrayXd W1;
// for (int lp=0; lp<4; lp++) {
//   cout<<"lp norm : "<<lp<<endl;
//   W1 = P->col_lpnorm(lp);
//   cout<<"Column lp-norm : "<<W1.minCoeff()<<", "<<W1.maxCoeff()<<endl;
//   W1 = P->row_lpnorm(lp);
//   cout<<"Row lp-norm : "<<W1.minCoeff()<<", "<<W1.maxCoeff()<<endl;
// }
