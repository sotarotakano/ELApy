// for matrices
#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// for vector
#include <vector>

#include <stdio.h>
#include <string>
#include <iostream>
#include <random>
#include <thread>
#include <future>

namespace py = pybind11;


//////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////
//              Energy landscape analysis               //
//////////////////////////////////////////////////////////

// -- Functions for Energy landscape analysis 

// -- Community Energy
inline double cEnergy(const arma::rowvec& state, const arma::rowvec& alpha, const arma::mat& beta){
  arma::mat res = -state * alpha.t() - (state* (state * beta).t() ) / 2;
  return as_scalar(res);
}


// Convert to decimal
unsigned long long convDec(arma::rowvec v)
{
  int tmp;
  std::string str;
  for(arma::uword i=0; i< v.n_elem; i++){
    tmp=v(i);
    str+=std::to_string(tmp);
  }
  return std::stoull(str, nullptr, 2); 
}

//////////////////////////////////////////////////////////
  
py::array_t<double> SteepestDescent_cpp_ind(const py::array_t<double>& params_arr){
  
  // ================================ //
  int term=0; 
  arma::mat energi;
  
  arma::mat params = carma::arr_to_mat<double>(params_arr);
  arma::rowvec alpha = params.row(params.n_rows-1);
  arma::mat beta = params.submat(0,0,params.n_rows-2,params.n_cols-1);

  arma::imat intmat=arma::randi(1, beta.n_cols, arma::distr_param(0, 1));
  arma::rowvec state=arma::conv_to<arma::rowvec>::from(intmat);

  arma::mat y1 = arma::conv_to<arma::mat>::from(state);

  double y2 = cEnergy(state, alpha, beta); 
  
  arma::mat ystat;
  double yene;
  arma::mat sign;
  double minene;
  arma::uword mp;
  // ================================ //
  
  do{
    // ============================= //
    ystat= y1;
    yene = y2;
    // ============================= //
    
    // -- Energy
    energi = -alpha-ystat*beta;
    sign = (-2 * ystat +1);
    energi %=sign; energi +=yene;
    
    minene =energi.min() ;
    if( minene < yene ){
      mp = energi.index_min();
      ystat(0, mp) = abs(ystat(0, mp)-1);
      
      y1.swap(ystat);
      y2 = minene;
      
    }else{
      term+=1;
    }
  } while (term==0);

  //return arma::join_rows(y1, arma::mat(1,1).fill(y2),arma::mat(1,1).fill(ss_idx));
  
  unsigned long long ss_idx = convDec(arma::conv_to<arma::rowvec>::from(y1));
  arma::mat res = arma::join_rows(y1, arma::mat(1,1).fill(y2),arma::mat(1,1).fill(ss_idx));
  py::array_t<double> res_arr = carma::mat_to_arr<double>(res);
  return res_arr;
}


////////////////////////////////////////////////////////////
arma::mat SteepestDescent_cpp(arma::rowvec state, arma::rowvec alpha, arma::mat beta){
  
  // ================================ //
  int term=0; 
  arma::mat energi; 
  arma::mat y1 = arma::conv_to<arma::mat>::from(state);
  double y2 = cEnergy(state, alpha, beta); 
  
  arma::mat ystat;
  double yene;
  arma::mat sign;
  double minene;
  arma::uword mp;
  // ================================ //
  
  do{
    // ============================= //
    ystat= y1;
    yene = y2;
    // ============================= //
    
    // -- Energy
    energi = -alpha-ystat*beta;
    sign = (-2 * ystat +1);
    energi %=sign; energi +=yene;
    
    minene =energi.min() ;
    if( minene < yene ){
      mp = energi.index_min();
      ystat(0, mp) = abs(ystat(0, mp)-1);
      
      y1.swap(ystat);
      y2 = minene;
      
    }else{
      term+=1;
    }
  } while (term==0);

  //return arma::join_rows(y1, arma::mat(1,1).fill(y2),arma::mat(1,1).fill(ss_idx));
  
  unsigned long long ss_idx = convDec(arma::conv_to<arma::rowvec>::from(y1));
  return arma::join_rows(y1, arma::mat(1,1).fill(y2),arma::mat(1,1).fill(ss_idx));
  
}


py::array_t<double> SteepestDescent_cpp_python(const py::array_t<double>& params_arr){
  
  // ================================ //
  int term=0; 
  arma::mat energi;
  
  arma::mat params = carma::arr_to_mat<double>(params_arr);
  arma::rowvec state = params.row(params.n_rows-1);
  
  arma::rowvec alpha = params.row(params.n_rows-2);
  arma::mat beta = params.submat(0,0,params.n_rows-3,params.n_cols-1);

  arma::mat y1 = arma::conv_to<arma::mat>::from(state);
  double y2 = cEnergy(state, alpha, beta); 
  
  arma::mat ystat;
  double yene;
  arma::mat sign;
  double minene;
  arma::uword mp;
  // ================================ //
  
  do{
    // ============================= //
    ystat= y1;
    yene = y2;
    // ============================= //
    
    // -- Energy
    energi = -alpha-ystat*beta;
    sign = (-2 * ystat +1);
    energi %=sign; energi +=yene;
    
    minene =energi.min() ;
    if( minene < yene ){
      mp = energi.index_min();
      ystat(0, mp) = abs(ystat(0, mp)-1);
      
      y1.swap(ystat);
      y2 = minene;
      
    }else{
      term+=1;
    }
  } while (term==0);

  //return arma::join_rows(y1, arma::mat(1,1).fill(y2),arma::mat(1,1).fill(ss_idx));
  
  unsigned long long ss_idx = convDec(arma::conv_to<arma::rowvec>::from(y1));
  arma::mat res = arma::join_rows(y1, arma::mat(1,1).fill(y2),arma::mat(1,1).fill(ss_idx));
  py::array_t<double> res_arr = carma::mat_to_arr<double>(res);
  return res_arr;
}



//////////////////////////////////////////////////////////
py::array_t<double> FindingTippingpoint_cpp_ind(const py::array_t<double>& params_arr,
                                                int tmax=10000){
  
  // ======================================= //
  // Distance between stable states
  arma::mat params = carma::arr_to_mat<double>(params_arr);
  arma::rowvec s2 = params.row(params.n_rows-1);
  arma::rowvec s1 = params.row(params.n_rows-2);
  arma::rowvec alpha = params.row(params.n_rows-3);
  arma::mat jj = params.submat(0,0,params.n_rows-4,params.n_cols-1);

  arma::uvec pd=arma::find(arma::abs(s1-s2)==1);
  arma::irowvec sequ=arma::conv_to<arma::irowvec>::from(arma::shuffle(pd));
  arma::rowvec tipState(alpha.n_elem+1);

  const int& sn=sequ.n_elem;

  // ======================================= //
  // ||||||||||||||||||||||||||||||||||| //
  // -- Definition
  double bde=arma::datum::inf;
  int tt=0 ;
  double minbde=arma::datum::inf;
  arma::uvec rpl;
  arma::irowvec seqnew;
  double bdenew;
  const int& SMrow=sn+1;
  const int& SMcol=s1.n_elem;
  arma::vec::fixed<2> c; 
  arma:: mat samplePath;
  int r;
  int column;
  
  arma::mat samplePathtmp=arma::zeros(SMrow, SMcol);
  samplePathtmp.row(0)=s1;
  const arma::uvec& v = arma::linspace<arma::uvec>(1,SMrow-1, SMrow-1);
  arma::vec CE=arma::vec(SMrow);
  
  // ||||||||||||||||||||||||||||||||||| //
  do{
    tt+=1;
    // ||||||||||||||||||||||||||||||||||| //
    seqnew=sequ;
    rpl = arma::randperm(sequ.n_elem, 2);
    
    seqnew.swap_cols(rpl(0), rpl(1));
    
    samplePath=samplePathtmp;
    
    // ||||||||||||||||||||||||||||||||||| //
    r=1;
    
    for(int l=0; l<sn; ++l){
      column=seqnew(l);
      
      for(int i=r; i<SMrow; ++i){
  
        samplePath(i, column)=1;
        
      }
      r+=1;
    }
      
    samplePath.each_row(v) -= s1;
    samplePath=abs(samplePath);
    
    // ||||||||||||||||||||||||||||||||||| //
    
    for(int i=0; i<SMrow; ++i){
      
      CE(i)=cEnergy(samplePath.row(i), alpha, jj);
      
    }
    // ||||||||||||||||||||||||||||||||||| //
    
    bdenew=CE.max();
    if(bdenew<minbde){
      minbde=std::move(bdenew);
      
      tipState.cols(0, SMcol-1)=samplePath.row(CE.index_max());
      tipState.col(tipState.n_cols-1)=minbde;
      
    }
    
    c(0)=1; c(1)= ((std::exp(bde))/(std::exp(bdenew)));
    if(arma::randu<double>() < c.min()){
      sequ.swap(seqnew);
      bde=std::move(bdenew);
    }
    
    // ||||||||||||||||||||||||||||||||||| //
  } while (tt<tmax);
  unsigned long long ss1_idx = convDec(arma::conv_to<arma::rowvec>::from(s1));
  unsigned long long ss2_idx = convDec(arma::conv_to<arma::rowvec>::from(s2));

  arma::mat tipStatemat = arma::conv_to<arma::mat>::from(tipState);
  arma::mat res=arma::join_rows(tipStatemat, arma::mat(1,1).fill(ss1_idx),arma::mat(1,1).fill(ss2_idx));
  py::array_t<double> res_arr = carma::mat_to_arr<double>(res);
  return res_arr;
}


arma::rowvec FindingTippingpoint_cpp(arma::rowvec s1,arma::rowvec s2, 
                                     arma::rowvec alpha, arma::mat jj,
                                     int tmax=10000){
  
  // ======================================= //
  // Distance between stable states
  arma::uvec pd=arma::find(arma::abs(s1-s2)==1);
  arma::irowvec sequ=arma::conv_to<arma::irowvec>::from(arma::shuffle(pd));
  arma::rowvec tipState(alpha.n_elem+1);

  const int& sn=sequ.n_elem;

  // ======================================= //
  // ||||||||||||||||||||||||||||||||||| //
  // -- Definition
  double bde=arma::datum::inf;
  int tt=0 ;
  double minbde=arma::datum::inf;
  arma::uvec rpl;
  arma::irowvec seqnew;
  double bdenew;
  const int& SMrow=sn+1;
  const int& SMcol=s1.n_elem;
  arma::vec::fixed<2> c; 
  arma:: mat samplePath;
  int r;
  int column;
  
  arma::mat samplePathtmp=arma::zeros(SMrow, SMcol);
  samplePathtmp.row(0)=s1;
  const arma::uvec& v = arma::linspace<arma::uvec>(1,SMrow-1, SMrow-1);
  arma::vec CE=arma::vec(SMrow);
  
  tmax = std::min(sn*(sn-1)/2+1,tmax);

  if (sn > 1){
    // ||||||||||||||||||||||||||||||||||| //
    do{
      tt+=1;
      // ||||||||||||||||||||||||||||||||||| //
      seqnew=sequ;
      rpl = arma::randperm(sequ.n_elem, 2);
      
      seqnew.swap_cols(rpl(0), rpl(1));
      
      samplePath=samplePathtmp;
      
      // ||||||||||||||||||||||||||||||||||| //
      r=1;
      
      for(int l=0; l<sn; ++l){
        column=seqnew(l);
        
        for(int i=r; i<SMrow; ++i){
    
          samplePath(i, column)=1;
          
        }
        r+=1;
      }
        
      samplePath.each_row(v) -= s1;
      samplePath=abs(samplePath);
      
      // ||||||||||||||||||||||||||||||||||| //
      
      for(int i=0; i<SMrow; ++i){
        
        CE(i)=cEnergy(samplePath.row(i), alpha, jj);
        
      }
      // ||||||||||||||||||||||||||||||||||| //
      
      bdenew=CE.max();
      if(bdenew<minbde){
        minbde=std::move(bdenew);
        
        tipState.cols(0, SMcol-1)=samplePath.row(CE.index_max());
        tipState.col(tipState.n_cols-1)=minbde;
        
      }
      
      c(0)=1; c(1)= ((std::exp(bde))/(std::exp(bdenew)));
      if(arma::randu<double>() < c.min()){
        sequ.swap(seqnew);
        bde=std::move(bdenew);
      }
      
      // ||||||||||||||||||||||||||||||||||| //
    } while (tt<tmax);
    
  }else{
    samplePath=samplePathtmp;
    tipState.cols(0, SMcol-1)=samplePath.row(0);
    tipState.col(tipState.n_cols-1)= cEnergy(samplePath.row(0), alpha, jj);
  }
  unsigned long long ss1_idx = convDec(arma::conv_to<arma::rowvec>::from(s1));
  unsigned long long ss2_idx = convDec(arma::conv_to<arma::rowvec>::from(s2));

  arma::mat tipStatemat = arma::conv_to<arma::mat>::from(tipState);
  return arma::join_rows(tipStatemat, arma::mat(1,1).fill(ss1_idx),arma::mat(1,1).fill(ss2_idx));
  //return tipState;
}


//////////////////////////////////////////////////////////

// table function similar to Rcpp
template<typename T>
std::unordered_map<T, int> tablecpp(const std::vector<T>& data){
    std::unordered_map<T, int> frequency;
    for (const auto& item : data){
        frequency[item]++;
    }
    return frequency;
}


//randomly pick out a integer from an uniform distribution
int rand_uniform_int(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);

    return dis(gen);
}

// log model for simple SA
inline arma::mat LogmodelSimplecpp(arma::mat m_mat, const arma::mat& a_mat, const arma::mat& b_mat){
  m_mat *= b_mat;
  const arma::mat& res_mat = 1/(exp(-a_mat - m_mat.each_row())+1);
  
  return res_mat;
}


// One step "Heat Bath Sampling"
arma::mat OnestepHBScpp(arma::mat y_mat, const arma::mat& lm_mat) {
  
  // Preparation // 
  const int& itr1=y_mat.n_cols;
  const int& itr2=y_mat.n_rows;
  
  for(int r=0; r < itr2; ++r){
    
    // Initial state //
    int uni=rand_uniform_int(0, itr1-1);
    double ne=lm_mat(r, uni);
    
    double randomNum=arma::randu<double>();
    
    // adjacency state//
    if(ne > randomNum){
      y_mat(r, uni) = 1;
    }else{
      y_mat(r, uni) = 0;
    }
    
  }
  return y_mat;
}


// To stop calculation.
int checkIdent(const arma::mat& ss, const arma::rowvec& ysim)
{
  int rows=ss.n_rows;
  arma::vec res(rows);
  
  for(int l=0; l<rows; l++){
    res(l)=all(ysim==ss.row(l));
  }
  
  return any(res);
}

// Convert to string
inline std::string convStr(const arma::rowvec& v)  {
  int tmp;
  std::string str;
  for(arma::uword i=0; i< v.n_elem; i++){
    tmp=v(i);
    str+=std::to_string(tmp);
  }
  
  return str; 
  
}

// Entropy numeric
inline double entropy(const arma::vec& v)
{
  arma::vec uniq=arma::unique(v);
  int N=uniq.n_elem;
  const double& total=v.n_elem;
  double ent=0;
  
  for(int i=0; i<N; i++){
    const double& prob=arma::sum(v==uniq(i))/total;
    ent+=prob*std::log2(prob);
  }
  // Rcout << probV << endl;
  
  return ent*(-1);
}

// Entropy string
inline double entropy2(const std::vector<std::string>& v)
{
  std::unordered_map<std::string,int> tab=tablecpp(v);
  const double& total=v.size();
  //NumericVector numtab=as<NumericVector>(tab);
  
  double ent=0;
  for(const auto& pair : tab){
    const double& prob=pair.second/total;
    ent+=prob*std::log2(prob);
  }
  
  return ent*(-1);
}


///////////////////////////////////////////////////////////////////
//- Calling function from the python (estimation of stable states)

py::array_t<double> SSestimate_cpp(const py::array_t<double>& alpha_arr, const py::array_t<double>& beta_arr,
                                   int itr=20000){
  
  arma::rowvec alpha = carma::arr_to_col<double>(alpha_arr).t();
  arma::mat beta = carma::arr_to_mat<double>(beta_arr);

  arma::mat res = arma::zeros(itr, beta.n_cols+2);
  arma::imat intmat;
  arma::rowvec state;
  arma::mat ss;
  
  for(int i=0; i<itr; ++i ){
    
    intmat=arma::randi(1, beta.n_cols, arma::distr_param(0, 1));
    state=arma::conv_to<arma::rowvec>::from(intmat);
    ss = SteepestDescent_cpp(state, alpha, beta);
    
    res.row(i) = ss;
  }
  //py::array_t<double> res_arr = carma::mat_to_arr<double>(res);
  //return res_arr;
  arma::colvec ssid = res.col(res.n_cols-1);
  arma::uvec unique_idxs=arma::find_unique(ssid);
  arma::mat minsets = res.rows(unique_idxs);

  py::array_t<double> minsets_arr = carma::mat_to_arr<double>(minsets);
  return minsets_arr;
}


//- Calling function from the python (estimation of tipping points)
py::array_t<double> TPestimate_cpp2(const py::array_t<double> comb_arr, 
                                    const py::array_t<double> minsets_arr,
                                    py::array_t<double> alpha_arr, py::array_t<double> beta_arr, 
                                    const int& tmax=10000){

  arma::mat comb = carma::arr_to_mat<double>(comb_arr);
  arma::mat minset = carma::arr_to_mat<double>(minsets_arr);

  arma::rowvec alpha = carma::arr_to_col<double>(alpha_arr).t();
  arma::mat beta = carma::arr_to_mat<double>(beta_arr);
  const int& resrow=comb.n_rows;
  arma::mat res = arma::zeros(resrow, beta.n_cols+3);
  
  int n; int m;
  arma::rowvec ss1; arma::rowvec ss2; arma::rowvec tp;
  
  for(int i=0; i<resrow; ++i ){
    n=comb(i, 0);
    m=comb(i, 1);
    
    ss1=minset.row(n);
    ss2=minset.row(m);
      
    tp = FindingTippingpoint_cpp(ss1, ss2, alpha, beta, tmax);
    
    res.row(i) = tp;
  }
  py::array_t<double> res_arr = carma::mat_to_arr<double>(res);
  return res_arr;
}


py::array_t<double> TPestimate_cpp(const py::array_t<double> & comb1_arr, 
                                   const py::array_t<double> & comb2_arr,
                                   py::array_t<double> alpha_arr, py::array_t<double> beta_arr, 
                                   const int& tmax=10000){

  arma::mat comb1 = carma::arr_to_mat<double>(comb1_arr);
  arma::mat comb2 = carma::arr_to_mat<double>(comb2_arr);
  arma::rowvec alpha = carma::arr_to_col<double>(alpha_arr).t();
  arma::mat beta = carma::arr_to_mat<double>(beta_arr);

  const int& resrow=comb1.n_rows;
  arma::mat res = arma::zeros(resrow, beta.n_cols+3);
  
  arma::rowvec ss1; arma::rowvec ss2; arma::rowvec tp;
  
  for(int i=0; i<resrow; ++i ){

    ss1=comb1.row(i);
    ss2=comb2.row(i);
      
    tp = FindingTippingpoint_cpp(ss1, ss2, alpha, beta, tmax);
    
    res.row(i) = tp;
  }
  py::array_t<double> res_arr = carma::mat_to_arr<double>(res);
  return res_arr;
}



//- Calling function from the python (estimation of stable states entropy)

py::array_t<double> SSentropy_cpp(py::array_t<double> uoc_arr, py::array_t<double> ss_arr,
                                  py::array_t<double> alpha_arr, py::array_t<double> beta_arr, 
                                  int seitr=1000, int convTime=10000){
  
  arma::mat uoc = carma::arr_to_mat<double>(uoc_arr);
  arma::mat ss = carma::arr_to_mat<double>(ss_arr);
  arma::rowvec alpha = carma::arr_to_col<double>(alpha_arr);
  arma::mat beta = carma::arr_to_mat<double>(beta_arr);
  
  arma::mat entropyres=arma::zeros(uoc.n_rows, 3);
  for(arma::uword i=0; i < uoc.n_rows; i++){
    
    arma::mat logmat;
    arma::rowvec ysim=uoc.row(i);
    int tt=0;
    std::vector<std::string> ssid(seitr);
    arma::mat stable=arma::zeros(seitr, 2);
    
    for(int l=0; l < seitr; l++){
      
      do{
        tt+=1;
        logmat=LogmodelSimplecpp(ysim, alpha, beta);
        ysim=OnestepHBScpp(ysim, logmat);
        
        if(checkIdent(ss, ysim)==1){
          break;
        }
      } while ( tt<convTime );
      
      ssid[l]=convStr(ysim);
      stable(l, 0)=tt;
      stable(l, 1)=((tt+1)==convTime);
      
    }
    
    entropyres(i,0)=entropy2(ssid);
    entropyres(i,1)=arma::mean(stable.col(0));
    entropyres(i,2)=arma::sum(stable.col(1));
  }
  py::array_t<double> entropyres_arr = carma::mat_to_arr<double>(entropyres);
  return entropyres_arr;
}


PYBIND11_MODULE(ELA, m) {
    m.doc() = "ELA functions in cpp";
    m.def("SSestimate_cpp", &SSestimate_cpp, "estimating stable states");
    m.def("SteepestDescent_cpp_ind", &SteepestDescent_cpp_ind, "calling steepest descent function directly");
    m.def("SteepestDescent_cpp_python", &SteepestDescent_cpp_python, "calling steepest descent function directly from python");
    m.def("FindingTippingpoint_cpp_ind", &FindingTippingpoint_cpp_ind, "calling steepest descent function directly");
    m.def("TPestimate_cpp", &TPestimate_cpp, "estimating tipping points");
    m.def("SSentropy_cpp", &SSentropy_cpp, "calculating entropy from a given set of stable states");
}