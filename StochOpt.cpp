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

namespace py = pybind11;

//randomly pick out a integer from an uniform distribution
int rand_uniform_int(int min, int max) {
    static std::mt19937 gen(std::random_device{}()); 
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);

    return dis(gen);
}

// log model for simple SA
inline arma::mat LogmodelSimplecpp(arma::mat m_mat, const arma::mat& a_mat, const arma::mat& b_mat){
  m_mat *= b_mat;
  const arma::mat& res_mat = 1/(exp(-a_mat - m_mat.each_row())+1);
  
  return res_mat;
}

// log model for full SA
inline arma::mat Logmodelcpp(arma::mat m,  const arma::mat& alpha, const arma::mat& beta){
  
  m *= beta;
  const arma::mat& res = 1/(exp(-alpha - m)+1);
  
  return res;
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

// Log Prior function
inline arma::mat Logprior(const arma::mat& alpha, const double& x) {
  
  arma::mat y = arma::zeros(alpha.n_rows, alpha.n_cols);
  y.fill(x);
  arma::mat lp = -arma::tanh((alpha/y)/2)/y;
  return lp;
}

//////////////////////////////////////////////////////////
//========  Main fucntion of the optimization ===========//
//========  only Inplicit environmetanl data  ===========//

py::array_t<double> simpleSA_cpp(const py::array_t<double>& ocData_arr, const double& we=0.001,
const int& totalit=1000,const double& lambda=0.01, const int& intv=100, bool runadamW=true, bool sparse=true){
  // ========================================= //
  // -- Initial parameter setting
  arma::mat ocData = carma::arr_to_mat<double>(ocData_arr);
  //const double& maxInt=50000;
  //const double& momentum=0.3;
  const double& nlocation = ocData.n_rows;
  const double& nspecies = ocData.n_cols;
  double learningrate0 = 0.05;

  arma::mat ystats=arma::trans(ocData);
  ystats *= ocData;
  arma::mat ysim = arma::zeros(nlocation, nspecies);
  const arma::mat& lp = (arma::sum(ocData, 0)+1)/(nlocation+2);
  arma::mat alphas = arma::log(lp/(1-lp));
  arma::mat beta = arma::zeros(nspecies, nspecies);
  arma::mat delalphas=arma::zeros(1, nspecies);
  arma::mat delbeta = arma::zeros(nspecies, nspecies);
  arma::mat betagrad;
  arma::mat alphasgrad;
  arma::mat logmat; 
  arma::mat ydif;
  arma::mat grad;
  arma::mat paramsmon;

  if (we < 0.00000001){
    runadamW = false;
  };

  if (lambda < 0.00000001){
    sparse = false;
  }

  if(runadamW){
    learningrate0=0.01;
  }

  const double& beta1 = 0.5;
  const double& beta2 = 0.99;
  int total_tt = 0;
  double beta1t = 1;
  double beta2t = 1;
  double minu = std::numeric_limits<double>::infinity();
  double uu = 0;
  int totalrows = nspecies*std::floor(totalit/intv);
  int updidx = 0;

  arma::mat vt = arma::zeros(nspecies,nspecies);
  arma::mat st = arma::zeros(nspecies,nspecies);
  arma::mat wt = arma::zeros(nspecies,nspecies);
  arma::mat qi = arma::zeros(nspecies,nspecies);
  arma::mat vth, sth, pos_wt, neg_wt, zz;
  
  
  arma::mat betaconst=(nlocation * arma::abs(arma::eye(nspecies, nspecies)-1));
  arma::rowvec asconst=(arma::mat(1, nspecies).fill(nlocation));
  std::vector<double> upds(totalrows);
  std::vector<double> points(totalrows);

  double upd;
  const double& momtm=0.3; //
  //double upcrit;
  // ========================================= //
  // Main part
  //double learningrate=0.1;
  for(int tt=0; tt < totalit; ++tt){
    // const double& momtm=0.9*(1-1/(0.1*tt+2));
    // -- Preference for co-occurence of species i and j
    logmat=LogmodelSimplecpp(ysim, alphas, beta);
    ysim=OnestepHBScpp(ysim, logmat);
    //mat ysimstats=trans(ysim) * ysim;
    ydif = ystats - (arma::trans(ysim) * ysim);
    if(runadamW){
      grad = ydif / betaconst;
      vt = beta1*vt + (1 - beta1)*grad;
      st = beta2*st + (1 - beta2)*(arma::square(grad));
      beta1t = beta1t*beta1; 
      beta2t = beta2t*beta2;
      vth = vt*(1/(1 - beta1t));
      sth = st*(1/(1 - beta2t));
      wt = wt + learningrate0*((vth/(arma::sqrt(sth) + 0.00000001))-we*wt);
      if(sparse){
        uu += learningrate0*10*lambda/nspecies;
        zz = wt;
        pos_wt = wt - (uu + qi);
        pos_wt.elem(arma::find(pos_wt < 0.0)).zeros();
        neg_wt = wt + (uu - qi);
        neg_wt.elem(arma::find(neg_wt > 0.0)).zeros();
        wt.elem(arma::find(zz < 0.0)) = neg_wt.elem(arma::find(zz < 0.0));
        wt.elem(arma::find(zz > 0.0)) = pos_wt.elem(arma::find(zz > 0.0));
        wt.diag() = zz.diag();
        qi += wt - zz;
      }
      arma::mat alphas = arma::diagvec(wt);
      beta = wt;
      beta.diag().fill(0);
    }else{
      // -- Beta gradient
      const double& learningrate=learningrate0;
      //const double& learningrate=learningrate0*1000/(998+1+tt);
      //betagrad = (ydif + Logprior(beta, 0.5)) / betaconst ;
      betagrad = ydif / betaconst;
      betagrad.diag().fill(0);
      // -- Alpha gradient
      //alphasgrad = (ydif.diag().t() + Logprior(alphas, 2))/asconst;
      alphasgrad = ydif.diag().t()/asconst;
      // -- delta
      betagrad %= arma::mat(nspecies, nspecies).fill((1-momtm)*learningrate);
      delbeta %= arma::mat(nspecies, nspecies).fill(momtm);
      delbeta += betagrad;
      alphasgrad %= arma::mat(1, nspecies).fill((1-momtm)*learningrate) ;
      delalphas %= arma::mat(1, nspecies).fill(momtm);
      delalphas += alphasgrad ;
      beta+=delbeta;
      alphas+=delalphas;
      if(sparse){
        uu += learningrate*lambda/nspecies;
        zz = beta;
        pos_wt = beta - (uu + qi);
        pos_wt.elem(arma::find(pos_wt < 0.0)).zeros();
        neg_wt = beta + (uu - qi);
        neg_wt.elem(arma::find(neg_wt > 0.0)).zeros();
        beta.elem(arma::find(zz < 0.0)) = neg_wt.elem(arma::find(zz < 0.0));
        beta.elem(arma::find(zz > 0.0)) = pos_wt.elem(arma::find(zz > 0.0));
        qi += beta - zz;
      }
    }
    // -- save intermediate status
    // max elementwise (delalpahs, delbeta)
    //upd = std::max(arma::max(arma::max(delalphas)),arma::max(arma::max(delbeta)));
    if (tt == 0 || rand_uniform_int(0,totalit) < totalrows*1.05){
      if (updidx < totalrows){
        upds[updidx] = arma::mean(arma::vectorise(arma::abs(ydif)/nlocation));
        points[updidx] = tt;
        updidx += 1;
      }
    }

    if (tt%intv == 1){
      //arma::rowvec upds = arma::ones(1,nspecies)*arma::mean(arma::vectorise(arma::abs(ydif)/nlocation));
      paramsmon = arma::join_cols(paramsmon,arma::join_rows(alphas.t(),beta.t()));
      
    }
    //upds.push_back(arma::mean(arma::vectorise(arma::abs(ydif)/nlocation)));
    
    // older version of stop criterion
    //  if (upd < minu) {
    //      minu = upd; 
    //      intv = 0;       
    //  } else {
    //      intv++;         
    //  }
    //
    //  if(intv > maxintv){
    //    total_tt = tt;
    //     //std::cout << "iterations: " << tt << std::endl;
    //    break;
    //  }

    // update upds (max len = 1000)
    //if(upds.n_rows >= 1000){
    //  upds.shed_row(0);
    //}
    //upds.insert_rows(upds.n_rows,1);
    //upds(upds.n_rows -1) = upd;
    // calc upcrit (roll diff mean)
    //if(upds.n_rows >= 2){
    //  upcrit = arma::mean(-arma::diff(upds));
    //}else{
    //  upcrit = upds[0];
    //}
    // stop iteration
    //  if(tt > 100 && (upcrit < qTh)){
        //std::cout << "iterations: " << tt << std::endl;
    //    break;
    //  }
        
    }
    upds[totalrows-1] = arma::mean(arma::vectorise(arma::abs(ydif)/nlocation));
    points[totalrows-1] = totalit-1;
    // ========================================= //
    //arma::rowvec iterations=arma::ones(1,nspecies)*total_tt;
    //arma::mat ystatsList=arma::join_rows(alphas.t(),beta.t(),iterations.t());
    arma::colvec upd_arma = arma::conv_to< arma::colvec >::from(upds);
    arma::colvec point_arma = arma::conv_to< arma::colvec >::from(points);
    arma::mat ystatsList=arma::join_rows(paramsmon,upd_arma,point_arma);
    
    //arma::mat ystatsList = paramsmon;
    py::array_t<double> ystats_arr = carma::mat_to_arr<double>(ystatsList);
    return ystats_arr;
}


void simpleSA_cpp_void(const py::array_t<double>& ocData_arr,const std::string& filename, 
                       const double& qTh=0.00001){
  // ========================================= //
  // -- Initial parameter setting
  arma::mat ocData = carma::arr_to_mat<double>(ocData_arr);
  const double& maxInt=50000;
  //const double& momentum=0.3;
  const double& nlocation = ocData.n_rows;
  const double& nspecies = ocData.n_cols;

  arma::mat ystats=arma::trans(ocData);
  ystats *= ocData;
  arma::mat ysim = arma::zeros(nlocation, nspecies);
  const arma::mat& lp = (arma::sum(ocData, 0)+1)/(nlocation+2);
  arma::mat alphas = arma::log(lp/(1-lp));
  arma::mat beta = arma::zeros(nspecies, nspecies);
  arma::mat delalphas=arma::zeros(1, nspecies);
  arma::mat delbeta = arma::zeros(nspecies, nspecies);
  arma::mat betagrad;
  arma::mat alphasgrad;
  arma::mat logmat; 
  arma::mat ydif;

  const double& learningrate0=0.1;
  arma::mat betaconst=(nlocation * arma::abs(arma::eye(nspecies, nspecies)-1));
  arma::rowvec asconst=(arma::mat(1, nspecies).fill(nlocation));
  arma::vec upds= {};
  double upd, upcrit;
  // ========================================= //
  // Main part
  //double learningrate=0.1;
  for(int tt=0; tt < maxInt; ++tt){
    const double& learningrate=learningrate0*1000/(998+1+tt);
    // const double& momtm=0.9*(1-1/(0.1*tt+2));
    const double& momtm=0.3; //
    // -- Preference for co-occurence of species i and j
    logmat=LogmodelSimplecpp(ysim, alphas, beta);
    ysim=OnestepHBScpp(ysim, logmat);
    //mat ysimstats=trans(ysim) * ysim;
    ydif = ystats - (arma::trans(ysim) * ysim);
    // -- Beta gradient
    betagrad = (ydif + Logprior(beta, 0.5)) / betaconst ;
    betagrad.diag().fill(0);
    // -- Alpha gradient
    alphasgrad = (ydif.diag().t() + Logprior(alphas, 2))/asconst;
    // -- delta
    betagrad %= arma::mat(nspecies, nspecies).fill((1-momtm)*learningrate);
    delbeta %= arma::mat(nspecies, nspecies).fill(momtm);
    delbeta += betagrad;
    alphasgrad %= arma::mat(1, nspecies).fill((1-momtm)*learningrate) ;
    delalphas %= arma::mat(1, nspecies).fill(momtm);
    delalphas += alphasgrad ;
    beta+=delbeta;
    alphas+=delalphas;

    // -- stop iteration
    // max elementwise (delalpahs, delbeta)
    upd = std::max(arma::max(arma::max(delalphas)),
    arma::max(arma::max(delbeta)));
    // update upds (max len = 1000)
    if(upds.n_rows >= 1000){
      upds.shed_row(0);
    }
    upds.insert_rows(upds.n_rows,1);
    upds(upds.n_rows -1) = upd;
    // calc upcrit (roll diff mean)
    if(upds.n_rows >= 2){
      upcrit = arma::mean(-arma::diff(upds));
    }else{
      upcrit = upds[0];
    }
    // stop iteration
      if(tt > 100 && (upcrit < qTh)){
        //std::cout << "iterations: " << tt << std::endl;
        break;
      }
        
    }
    // ========================================= //
    //arma::rowvec ydif_sum= arma::abs(arma::sum(ydif,0));
    //arma::mat ystatsList=arma::join_rows(alphas.t(),beta.t(),ydif_sum.t());
    arma::mat ystatsList=arma::join_rows(alphas.t(),beta.t());
    ystatsList.save(filename);
    //ystatsList.save(filename,arma::csv_ascii);
}


//========  with Explicit environmetanl data  ===========//
py::array_t<double> fullSA_cpp(const py::array_t<double>& ocData_arr, const py::array_t<double>& envData_arr,
                               const double& we=0.001,const int& totalit=1000,const double& lambda=0.01, 
                               const int& intv=100, bool runadamW=true, bool sparse=true){
                                 
  // ========================================= //
  // -- Initial parameter setting
  arma::mat ocData = carma::arr_to_mat<double>(ocData_arr);
  arma::mat envData = carma::arr_to_mat<double>(envData_arr);
  double learningrate0 = 0.05;
  //const double& momentum=0.3;
  const double& nlocation = ocData.n_rows;
  const double& nspecies = ocData.n_cols;
  const double& nenvironment = envData.n_cols;

  if (we < 0.00000001){
    runadamW = false;
  };

  if (lambda < 0.00000001){
    sparse = false;
  }

  if(runadamW){
    learningrate0=0.01;
  }

  // common parameters and variables
  arma::mat ystats = arma::trans(ocData) * ocData;
  arma::mat yenvstats = arma::trans(envData) * ocData;
  arma::mat ysim = arma::zeros(nlocation, nspecies);

  // Parameters and variables for normal SA
  const arma::mat& lp = (arma::sum(ocData, 0)+1)/(nlocation+2);
  arma::mat alphas = arma::log(lp/(1-lp));
  //arma::mat alphas = arma::zeros(1,nspecies);
  arma::mat beta = arma::zeros(nspecies, nspecies);
  arma::mat alphae = arma::zeros(nenvironment, nspecies);
  
  arma::mat delalphas= arma::zeros(1, nspecies); 
  arma::mat delalphae = arma::zeros(nenvironment, nspecies); 
  arma::mat delbeta = arma::zeros(nspecies, nspecies);
  
  arma::mat betagrad = arma::zeros(nspecies, nspecies);
  arma::mat alphasgrad = arma::zeros(1, nspecies);
  arma::mat alphaegrad = arma::zeros(nenvironment, nspecies);
  arma::mat alpha; arma::mat logmat; arma::mat ydif; arma::mat yenvdiff;
  
  arma::mat betaconst=(nlocation * arma::abs(arma::eye(nspecies, nspecies)-1));
  arma::rowvec asconst=(arma::mat(1, nspecies).fill(nlocation));
  arma::mat aeconst=(arma::mat(nenvironment, nspecies).fill(nlocation));
  
  // AdamW parameters and variables (occurence matrix)
  const double& obeta1 = 0.5; const double& obeta2 = 0.99;
  double obeta1t = 1; double obeta2t = 1;
  arma::mat ovt = arma::zeros(nspecies,nspecies);
  arma::mat ost = arma::zeros(nspecies,nspecies);
  arma::mat owt = arma::zeros(nspecies,nspecies);
  arma::mat ograd, ovth, osth;
  
  // AdamW parameters and variables (environmental matrix)
  const double& ebeta1 = 0.5; const double& ebeta2 = 0.99;
  double ebeta1t = 1; double ebeta2t = 1;
  arma::mat evt = arma::zeros(nenvironment,nspecies);
  arma::mat est = arma::zeros(nenvironment,nspecies);
  arma::mat ewt = arma::zeros(nenvironment,nspecies);
  arma::mat egrad,evth,esth;

  // Parameters for sparse 
  double minu = std::numeric_limits<double>::infinity();
  double uu = 0; 
  arma::mat qi = arma::zeros(nspecies,nspecies);
  arma::mat pos_owt, neg_owt, zz;
  
  // for storing intermediates
  int totalrows = nspecies*std::floor(totalit/intv);
  std::vector<double> upds(totalrows);
  std::vector<double> points(totalrows);
  int updidx = 0; 
  arma::mat paramsmon;

  const double& momtm=0.3; //
  // ========================================= //
  // Main part
  
  for(int tt=0; tt < totalit; ++tt){
    //const double& learningrate=learningrate0*1000/(998+1+tt);
    //const double& momtm=0.9*(1-1/(0.1*tt+2));
    
    // -- Preference for co-occurence of species i and j
    alpha=arma::mat(envData * alphae).each_row() + alphas;
    
    // -- 
    logmat=Logmodelcpp(ysim, alpha, beta);
    ysim=OnestepHBScpp(ysim, logmat);

    ydif = ystats - (arma::trans(ysim) * ysim);
    yenvdiff= yenvstats-(arma::trans(envData) * ysim);

    if(runadamW){
      // h and J (occurence matrix)
      ograd = ydif / betaconst;
      ovt = obeta1*ovt + (1 - obeta1)*ograd;
      ost = obeta2*ost + (1 - obeta2)*(arma::square(ograd));
      obeta1t = obeta1t*obeta1; 
      obeta2t = obeta2t*obeta2;
      ovth = ovt*(1/(1 - obeta1t));
      osth = ost*(1/(1 - obeta2t));
      owt = owt + learningrate0*((ovth/(arma::sqrt(osth) + 0.00000001))-we*owt);
      // e (environmental matrix)
      egrad = yenvdiff / aeconst;
      evt = ebeta1*evt + (1 - ebeta1)*egrad;
      est = ebeta2*est + (1 - ebeta2)*(arma::square(egrad));
      ebeta1t = ebeta1t*ebeta1; 
      ebeta2t = ebeta2t*ebeta2;
      evth = evt*(1/(1 - ebeta1t));
      esth = est*(1/(1 - ebeta2t));
      ewt = ewt + learningrate0*((evth/(arma::sqrt(esth) + 0.00000001)));

      if(sparse){
        uu += learningrate0*10*lambda/nspecies;
        zz = owt;
        pos_owt = owt - (uu + qi);
        pos_owt.elem(arma::find(pos_owt < 0.0)).zeros();
        neg_owt = owt + (uu - qi);
        neg_owt.elem(arma::find(neg_owt > 0.0)).zeros();
        owt.elem(arma::find(zz < 0.0)) = neg_owt.elem(arma::find(zz < 0.0));
        owt.elem(arma::find(zz > 0.0)) = pos_owt.elem(arma::find(zz > 0.0));
        owt.diag() = zz.diag();
        qi += owt - zz;
      }
      arma::mat alphas = arma::diagvec(owt);
      beta = owt;
      beta.diag().fill(0);
      alphae = ewt;

    }else{
      const double& learningrate=learningrate0;
      // -- Beta gradient 
      //betagrad = (ydif + Logprior(beta, 0.5)) / betaconst;
      betagrad = ydif / betaconst;
      betagrad.diag().fill(0);
      
      // -- Alpha gradient
      //alphasgrad = (ydif.diag().t() + Logprior(alphas, 2))/asconst;
      alphasgrad = (ydif.diag().t())/asconst;
      //alphaegrad = (yenvdiff+Logprior(alphae,2))/aeconst;
      alphaegrad = yenvdiff/aeconst;

      // -- delta beta
      betagrad %= arma::mat(nspecies, nspecies).fill((1-momtm)*learningrate);
      delbeta %= arma::mat(nspecies, nspecies).fill(momtm);
      delbeta += betagrad;
      
      // -- delta alphas
      alphasgrad %= arma::mat(1, nspecies).fill((1-momtm)*learningrate) ;
      delalphas %= arma::mat(1, nspecies).fill(momtm);
      delalphas+=alphasgrad ;   
      
      // -- delta alphae
      alphaegrad %= arma::mat(nenvironment, nspecies).fill((1-momtm)*learningrate);
      delalphae %= arma::mat(nenvironment, nspecies).fill(momtm);
      delalphae += alphaegrad;
      
      beta+=delbeta; 
      alphas+=delalphas; 
      alphae+=delalphae;
      if(sparse){
        uu += learningrate*lambda/nspecies;
        zz = beta;
        pos_owt = beta - (uu + qi);
        pos_owt.elem(arma::find(pos_owt < 0.0)).zeros();
        neg_owt = beta + (uu - qi);
        neg_owt.elem(arma::find(neg_owt > 0.0)).zeros();
        beta.elem(arma::find(zz < 0.0)) = neg_owt.elem(arma::find(zz < 0.0));
        beta.elem(arma::find(zz > 0.0)) = pos_owt.elem(arma::find(zz > 0.0));
        qi += beta - zz;
      }
    }


    // -- save intermediate status
    if (tt == 0 || rand_uniform_int(0,totalit) < totalrows*1.05){
      if (updidx < totalrows){
        upds[updidx] = arma::mean(arma::vectorise(arma::abs(ydif)/nlocation));
        points[updidx] = tt;
        updidx += 1;
      }
    }

    if (tt%intv == 1){
      paramsmon = arma::join_cols(paramsmon,arma::join_rows(alphas.t(),alphae.t(),beta.t()));
      
    }
    
  }
  // -- summarize results 
  upds[totalrows-1] = arma::mean(arma::vectorise(arma::abs(ydif)/nlocation));
  points[totalrows-1] = totalit-1;
  // ========================================= //
  arma::colvec upd_arma = arma::conv_to< arma::colvec >::from(upds);
  arma::colvec point_arma = arma::conv_to< arma::colvec >::from(points);
  arma::mat ystatsList=arma::join_rows(paramsmon,upd_arma,point_arma);
  py::array_t<double> ystats_arr = carma::mat_to_arr<double>(ystatsList);
  return ystats_arr;
}



PYBIND11_MODULE(StochOpt, m) {
    m.doc() = "StochOpt functions in cpp";
    m.def("simpleSA_cpp", &simpleSA_cpp, "simple SA optimization with only implicit environmental parameters");
    m.def("simpleSA_cpp_void", &simpleSA_cpp_void, "simple SA optimization with only implicit environmental parameters");
    m.def("fullSA_cpp", &fullSA_cpp, "full SA optimization with explicit environmental parameters");
}