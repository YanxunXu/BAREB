#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <R.h>
#include <Rmath.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <vector>
# include <cstdlib>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <cstring>
using namespace std;
//this one fix the update Beta, change one E(i) to i


using namespace Rcpp;
const double log_e = log(exp(1));
const double Inf = std::numeric_limits<double>::infinity();
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::rowvec nonnan_ind(arma::rowvec x){
  int n=x.n_cols;
  arma::rowvec ans(n);
  int flag = 0;
  for(int i=0;i<n; i++){
    if(x(i)==x(i)){
      ans(flag) = i;
      flag ++;
    }
  }
  ans.reshape(1,flag);
  return (ans);
}
// [[Rcpp::export]]
arma::mat rowsome(arma::mat x,
                  arma::rowvec ind){
  int C = x.n_cols;
  int R = ind.n_cols;
  arma::mat ans(R,C);
  for(int i=0; i<R; i++){
    ans.row(i) = x.row(ind(i));
  }
  return(ans);
}

// [[Rcpp::export]]
arma::rowvec ton(int n){
  //ton(4) should give (0,1,2,3)
  arma::rowvec ans(n);
  for(int i=0; i<n; i++){
    ans(i) = i;
  }
  return ans;
}

// [[Rcpp::export]]
arma::mat colsome(arma::mat x,
                  arma::rowvec ind){
  int R = x.n_rows;
  int C = ind.n_cols;
  arma::mat ans(R,C);
  for(int i=0; i<C; i++){
    ans.col(i) = x.col(ind(i));
  }
  return(ans);
}

// [[Rcpp::export]]
arma::mat subsome(arma::mat x,
                  arma::rowvec ind){
  int n = ind.n_cols;
  int m = x.n_rows;
  arma::mat temp(m, n);
  arma::mat ans(n,n);
  temp = rowsome(x, ind);
  ans = colsome(temp,ind);
  return (ans);
}


double vectornorm(arma::rowvec x){
  double ans=0;
  ans = sum(x%x);
  return ans;
}

// [[Rcpp::export]]
double dmvnrm_arma0(arma::rowvec x,
                    arma::rowvec mean,
                    double sigma_square,
                    bool logd = false) {
  
  double out = vectornorm(x-mean);
  out = out/(-2 * sigma_square);
  if (logd == false) {
    out = exp(out);
  }
  return(out);
}

// [[Rcpp::export]]
double dmvnrm_arma(arma::rowvec x,
                   arma::rowvec mean,
                   double sigma_square,
                   bool logd = false) {
  arma::rowvec non_ind = nonnan_ind(x);
  arma::rowvec xnew = colsome(x,non_ind);
  arma::rowvec meannew = colsome(mean, non_ind);
  
  if(xnew.n_cols == 0){
    if(logd){
      return (0);
    }
    else{
      return (1);
    }
  }
  
  return(dmvnrm_arma0(xnew,meannew,sigma_square,logd));
}


int rmunoim(arma::rowvec probs) {
  //this function returns a int, not a vector
  //rmultinom(1,1,probs)
  int k = probs.n_cols;
  IntegerVector ans(k);
  int r=0;
  rmultinom(1, probs.begin(), k, ans.begin());
  for(int i=0; i<k; i++){
    if(ans(i)==1){
      r=i+1;
    }
  }
  return r;
}

int sampleint(int n){
  arma::rowvec beta(n);
  arma::mat temp(1,1);
  temp = 1.0/n;
  beta = arma::repmat(temp,1,n);
  int ans = rmunoim(beta);
  return ans;
}



// [[Rcpp::export]]
double kernelC(arma::rowvec x, arma::rowvec y, double theta, double tau){
  double ans;
  ans = exp(-vectornorm(x-y)/(theta*theta));
  return ans;
}

arma::mat eye(int n){
  arma:: mat ans(n,n);
  ans.eye();
  return (ans);
}

// [[Rcpp::export]]
arma::mat updateC(arma::mat Z,double theta, double tau){
  int K=Z.n_rows;
  int L = Z.n_cols;
  int j;
  arma::rowvec q(K);
  arma::mat C = eye(K);
  arma::rowvec mean(L);
  arma::mat Temp(K,K);
  mean.fill(0);
  C = C*0.00001;
  
  for(int i=0;i<K;i++){
    q(i) = dmvnrm_arma0(Z.row(i),mean,tau*tau);
    j = K-1;
    while(j>=i){
      C(i,j) += kernelC(Z.row(i),Z.row(j),theta,tau);
      j--;
    }
    while(j>=0){
      C(i,j) = C(j,i);
      j--;
    }
  }
  Temp = q.t()*q;
  return (C % Temp);
  //return C;
}

// [[Rcpp::export]]
arma::rowvec getind(arma::rowvec x, double c){
  int n = x.n_cols;
  arma::rowvec ans(n);
  int flag = 0;
  for(int i=0; i<n; i++){
    if(x(i)==c){
      ans(flag) = i;
      flag++;
    }
  }
  ans.reshape(1,flag);
  return(ans);
}

arma::mat matrix(arma::rowvec x, int n){
  int m = x.n_cols;
  arma::mat ans(n,m);
  for(int i=0; i<n; i++){
    ans.row(i)=x;
  }
  return(ans);
}

arma::rowvec getgamma(arma::cube x, int i, int j){
  int n = x.n_cols;
  int m = x.n_rows;
  arma::mat temp(m,n);
  arma::rowvec ans(n);
  temp = x.slice(i);
  ans = temp.row(j);
  return(ans);
}

arma::mat minusvalue(arma::mat x, int j, arma::rowvec ind, arma::rowvec y){
  //used for Y.theta[i,indj]<-Y[i,indj]-xbeta
  int n=ind.n_cols;
  for(int i=0; i<n;i++){
    x(j,ind(i)) -= y(i);
  }
  return x;
}

double findT(arma::rowvec x){
  double ans= -1;
  int n = x.n_cols;
  for(int i=0; i<n; i++){
    if(x(i)>0){
      ans = i;
    }
  }
  return ans;
}




arma::mat removeii(arma::mat A, int a, int b){
  //a,b start from 1
  //remove ath row and bth column
  int row=A.n_rows;
  int col=A.n_cols;
  arma::mat ans(row-1, col-1);
  arma::mat temp (row-1, col);
  for(int i=0;i<row-1;i++){
    if(i<a-1){
      temp.row(i) = A.row(i);
    }
    else{
      temp.row(i) = A.row(i+1);
    }
  }
  for(int j=0; j<col-1; j++){
    if(j<b-1){
      ans.col(j) = temp.col(j);
    }
    else{
      ans.col(j) = temp.col(j+1);
    }
  }
  
  
  return ans;
}

arma::rowvec removei(arma::rowvec A, int a){
  //need use i+1, since a starts from 1
  int k = A.n_cols;
  arma::rowvec ans(k-1);
  for (int i=0; i<k-1;i++){
    if(i<a-1){
      ans(i) = A(i);
    }
    else{
      ans(i) = A(i+1);
    }
  }
  return ans;
}

arma::vec cDmu(arma::rowvec x, arma::vec y){
  arma::rowvec ans(x.n_cols);
  ans = x%(y.t());
  return(ans.t());
}

// [[Rcpp::export]]
arma::rowvec ind(arma::rowvec x, arma::rowvec index){
  int n = accu(index);
  arma::rowvec ans(n);
  int flag = 0;
  int i = 0;
  while(flag<n){
    if(index(i)){
      ans(flag) = x(i);
      flag ++;
    }
    i++;
  }
  return(ans);
}



// [[Rcpp::export]]
List updateBeta(arma::mat X,
                arma::mat Y,
                arma::mat Z,
                arma::mat delta,
                arma::mat Beta,
                arma::cube Gamma,
                arma::rowvec E,
                arma::mat R,
                double S,
                arma::rowvec Ds,
                arma::mat mustar,
                arma::mat mu,
                double sigma,
                arma::rowvec c,
                arma::mat C,
                arma::mat step,
                arma::mat runif,
                int n, int m, int T0, int p, int q, arma::mat D,
                double theta, double tau){
  List ans;
  arma::rowvec Betanew(p);
  arma::vec mustarnewj(T0);
  arma::vec munewj(m);
  arma::mat mustarnew(n,T0);
  arma::mat munew(n,m);
  arma::mat Ybeta(n,m);
  arma::rowvec indj(m);
  arma::mat Zj(m,q);
  arma::rowvec gammaij(q);
  arma::rowvec betai(p);
  arma::rowvec indi(n);
  arma::vec beta0(m);
  arma::mat Cminus(m-1,m-1);
  arma::rowvec b(S-1);
  arma::mat temp(1,1);
  arma::rowvec bnew(S-1);
  int l;
  double likelinow, likelinew, fnow, fnew,va;
  munew = mu;
  mustarnew = mustar;
  Ybeta = Y;
  for(int i=0; i<n; i++){
    for(int j=1; j<=Ds(E(i)-1);j++){
      l=getind(R.row(E(i)-1),j).n_cols;
      indj.set_size(1,l);
      indj = getind(R.row(E(i)-1),j);
      Zj.set_size(l,q);
      Zj = rowsome(Z,indj);
      gammaij = Gamma.slice(E(i)-1).row(j-1);
      Ybeta = minusvalue(Ybeta,i,indj, (Zj*gammaij.t()).t());
    }
  }
  //now update beta one by one row
  for(int i=0; i<S;i++){
    likelinow = 0;
    betai = Beta.row(i);
    l=getind(E,i+1).n_cols;
    indi.set_size(1,l);
    indi = getind(E,i+1);
    for(int j=0; j<l; j++){
      //j in R is indi(j) now
      beta0 = matrix(X.row(indi(j)),m) * betai.t();
      likelinow += dmvnrm_arma(Ybeta.row(indi(j)),  beta0.t(), sigma , true);
      for(int t=0; t<T0; t++){
        va = min(5.0, mustar(indi(j),t));
        va = max(va, -5.0);
        likelinow += R::pnorm(va,0,1,true, true)* delta(indi(j),t) + 
          log(1-R::pnorm(va,0,1,true, false))*(1-delta(indi(j),t));
      }
    }
    Cminus = removeii(C,i+1,i+1);
    b = removei(C.row(i),i+1);
    temp = b*inv(Cminus)*b.t();
    fnow = kernelC(Beta.row(i),Beta.row(i),theta,tau) -  temp(0,0);
    //update Beta one by one
    for(int j=0; j<p;j++){
      Betanew = Beta.row(i);
      Betanew(j) = Betanew(j) + step(i,j);
      likelinew = 0;
      for(int k=0; k<l;k++){
        //k in R is now indi(k)
        beta0 = matrix(X.row(indi(k)),m)*Betanew.t();
        
        likelinew += dmvnrm_arma(Ybeta.row(indi(k)),  beta0.t(),  sigma , true);
        munewj = mu.row(indi(k)).t() + matrix(X.row(indi(k)),m) * (Betanew.t()-Beta.row(i).t());
        mustarnewj = mustar.row(indi(k)).t() + cDmu(c,D*(munewj)) - cDmu(c,D*mu.row(indi(k)).t());
        mustarnew.row(indi(k)) = mustarnewj.t();
        munew.row(indi(k)) = munewj.t();
        for(int t=0; t<T0; t++){
          va = min(5.0, mustarnewj(t));
          va = max(va, -5.0);
          likelinew += R::pnorm(va,0,1,true,true)*delta(indi(k),t) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(indi(k),t));
        }
      }
      //compute prior
      for(int s=0; s<S; s++){
        if(s<i){
          bnew(s) = kernelC(Beta.row(s), Betanew, theta, tau);
        }
        else if(s>i){
          bnew(s-1) = kernelC(Beta.row(s), Betanew,theta,tau);
        }
      }
      temp = bnew*inv(Cminus)*bnew.t();
      fnew = kernelC(Betanew, Betanew,theta,tau) - temp(0,0);
      if(l==0){
        likelinew = 0;
        likelinow = 0;
      }
      //cout<<"i="<<i+1<<"j="<<j+1<<"likelinow="<<likelinow<<", likelinew="<<likelinew<<endl;
      if((likelinew - likelinow + log(fnew/fnow))>log(runif(i,j))){
        b = bnew;
        fnow = fnew;
        //update C
        for(int s=0; s<S; s++){
          if(s<i){
            C(i,s) = bnew(s);
            C(s,i) = bnew(s);
          }
          else if(s>i){
            C(i,s) = bnew(s-1);
            C(s,i) = bnew(s-1);
          }
          else{
            C(i,i) = kernelC(Betanew,Betanew,theta, tau);
          }
        }
        likelinow = likelinew;
        Beta.row(i) = Betanew;
        mustar = mustarnew;
        mu = munew;
      }
    }
  }
  ans["C"] = C;
  ans["Beta"] = Beta;
  ans["mu"]= mu;
  ans["mustar"] = mustar;
  ans["Ybeta"] = Ybeta;
  return (ans);
  
}



// [[Rcpp::export]]
List updateBetanomiss(arma::mat X,
                      arma::mat Y,
                      arma::mat Z,
                      arma::mat delta,
                      arma::mat Beta,
                      arma::cube Gamma,
                      arma::rowvec E,
                      arma::mat R,
                      double S,
                      arma::rowvec Ds,
                      arma::mat mu,
                      double sigma,
                      arma::mat C,
                      arma::mat step,
                      arma::mat runif,
                      int n, int m, int T0, int p, int q, arma::mat D,
                      double theta, double tau){
  List ans;
  arma::rowvec Betanew(p);
  arma::vec munewj(m);
  arma::mat munew(n,m);
  arma::mat Ybeta(n,m);
  arma::rowvec indj(m);
  arma::mat Zj(m,q);
  arma::rowvec gammaij(q);
  arma::rowvec betai(p);
  arma::rowvec indi(n);
  arma::vec beta0(m);
  arma::mat Cminus(m-1,m-1);
  arma::rowvec b(S-1);
  arma::mat temp(1,1);
  arma::rowvec bnew(S-1);
  int l;
  double likelinow, likelinew, fnow, fnew;
  munew = mu;
  Ybeta = Y;
  for(int i=0; i<n; i++){
    for(int j=1; j<=Ds(E(i)-1);j++){
      l=getind(R.row(E(i)-1),j).n_cols;
      indj.set_size(1,l);
      indj = getind(R.row(E(i)-1),j);
      Zj.set_size(l,q);
      Zj = rowsome(Z,indj);
      gammaij = Gamma.slice(E(i)-1).row(j-1);
      Ybeta = minusvalue(Ybeta,i,indj, (Zj*gammaij.t()).t());
    }
  }
  //now update beta one by one row
  for(int i=0; i<S;i++){
    likelinow = 0;
    betai = Beta.row(i);
    l=getind(E,i+1).n_cols;
    indi.set_size(1,l);
    indi = getind(E,i+1);
    for(int j=0; j<l; j++){
      //j in R is indi(j) now
      beta0 = matrix(X.row(indi(j)),m) * betai.t();
      likelinow += dmvnrm_arma(Ybeta.row(indi(j)),  beta0.t(), sigma , true);
    }
    Cminus = removeii(C,i+1,i+1);
    b = removei(C.row(i),i+1);
    temp = b*inv(Cminus)*b.t();
    fnow = kernelC(Beta.row(i),Beta.row(i),theta,tau) -  temp(0,0);
    //update Beta one by one
    for(int j=0; j<p;j++){
      Betanew = Beta.row(i);
      Betanew(j) = Betanew(j) + step(i,j);
      likelinew = 0;
      for(int k=0; k<l;k++){
        //k in R is now indi(k)
        beta0 = matrix(X.row(indi(k)),m)*Betanew.t();
        
        likelinew += dmvnrm_arma(Ybeta.row(indi(k)),  beta0.t(),  sigma , true);
        munewj = mu.row(indi(k)).t() + matrix(X.row(indi(k)),m) * (Betanew.t()-Beta.row(i).t());
        munew.row(indi(k)) = munewj.t();
      }
      //compute prior
      for(int s=0; s<S; s++){
        if(s<i){
          bnew(s) = kernelC(Beta.row(s), Betanew, theta, tau);
        }
        else if(s>i){
          bnew(s-1) = kernelC(Beta.row(s), Betanew,theta,tau);
        }
      }
      temp = bnew*inv(Cminus)*bnew.t();
      fnew = kernelC(Betanew, Betanew,theta,tau) - temp(0,0);
      if(l==0){
        likelinew = 0;
        likelinow = 0;
      }
      //cout<<"i="<<i+1<<"j="<<j+1<<"likelinow="<<likelinow<<", likelinew="<<likelinew<<endl;
      if((likelinew - likelinow + log(fnew/fnow))>log(runif(i,j))){
        b = bnew;
        fnow = fnew;
        //update C
        for(int s=0; s<S; s++){
          if(s<i){
            C(i,s) = bnew(s);
            C(s,i) = bnew(s);
          }
          else if(s>i){
            C(i,s) = bnew(s-1);
            C(s,i) = bnew(s-1);
          }
          else{
            C(i,i) = kernelC(Betanew,Betanew,theta, tau);
          }
        }
        likelinow = likelinew;
        Beta.row(i) = Betanew;
        mu = munew;
      }
    }
  }
  ans["C"] = C;
  ans["Beta"] = Beta;
  ans["mu"]= mu;
  return (ans);
  
}

int Nuni(arma::rowvec x){
  arma::rowvec a = unique(x);
  int n = a.n_cols;
  return (n);
}

arma::mat getGammai(arma::cube Gamma, int j, int ds, int q){
  arma::mat ans(ds, q);
  for(int i=0; i<ds;i++){
    ans.row(i) = getgamma(Gamma, j,i);
  }
  return(ans);
}

arma::mat putvalue(arma::mat x, int k, arma::rowvec indi, arma::rowvec value){
  int n = indi.n_cols;
  for(int i=0; i<n; i++){
    x(k,indi(i)) = value(i);
  }
  return (x);
}

arma::rowvec computemu(arma::mat mu, int k, arma::rowvec indi, arma::mat Zi, arma::rowvec gamma, arma::rowvec gammanew){
  int n = indi.n_cols;
  arma::rowvec ans(n);
  arma::vec temp(n);
  ans = colsome(mu.row(k), indi);
  temp = Zi *(gammanew.t()-gamma.t());
  ans += temp.t();
  return (ans);
}

arma::rowvec computemustar(arma::mat mustar, int k, arma::rowvec indT, arma::rowvec c,
                           arma::mat D, arma::rowvec muk, arma::rowvec muknew, int T0){
  int n = indT.n_cols;
  arma::vec temp(T0);
  arma::rowvec ans(n);
  ans = colsome(mustar.row(k), indT);
  temp = cDmu(c,D*muknew.t()) - cDmu(c, D*muk.t());
  ans += colsome(temp.t(), indT);
  return(ans);
}

arma::cube putGamma(arma::cube Gamma, arma::mat Gammai, int i, int q){
  arma::cube ans = Gamma;
  int R = Gamma.n_rows;
  int n = Gammai.n_rows;
  arma::mat temp(R,q);
  temp = ans.slice(i);
  for(int j=0; j<n; j++){
    temp.row(j) = Gammai.row(j);
  }
  ans.slice(i) = temp;
  return(ans);
}

// [[Rcpp::export]]
arma::rowvec ind1(arma::rowvec x, arma::rowvec indi, arma::rowvec index){
  //indi starts from 0
  int l = indi.n_cols;
  arma::rowvec ans(l);
  ans.fill(0);
  int flag = 0;
  for(int i=0; i<l; i++){
    if(index(indi(i))){
      ans(flag) = x(indi(i));
      flag++;
    }
  }
  ans.reshape(1,flag);
  return(ans);
}



arma::mat removeiii(arma::mat x, int a){
  //a starts from 1
  int R = x.n_rows-1;
  int C = x.n_cols;
  arma::mat ans(R,C);
  for(int i=0; i<R; i++){
    if(i<a-1){
      ans.row(i) = x.row(i);
    }
    else{
      ans.row(i) = x.row(i+1);
    }
  }
  return (ans);
}


// [[Rcpp::export]]
List updateGamma(arma::mat X,
                 arma::mat Y,
                 arma::mat Z,
                 arma::mat delta,
                 arma::mat Beta,
                 arma::cube Gamma,
                 arma::rowvec E,
                 arma::mat R,
                 double S,
                 arma::rowvec Ds,
                 arma::mat mu,
                 arma::mat mustar,
                 double sigma,
                 arma::rowvec c,
                 arma::cube step,
                 arma::cube runif,
                 int n, int m, int T0, int p, int q, arma::mat D,
                 double theta, double tau){
  List ans;
  arma::mat mustarnew(n, T0);
  arma::mat munew(n, m);
  arma::rowvec inds(n);
  arma::rowvec indi(m);
  arma::rowvec indT(T0);
  arma::mat Gammai(max(Ds),q);
  mustarnew = mustar;
  arma::rowvec temp(T0);
  munew =mu;
  arma::mat C(10,10);
  arma::mat Cminus(9,9);
  arma::mat temp1(1,1);
  arma::rowvec b(9);
  arma::rowvec temp2(10);
  arma::rowvec bnew(9);
  arma::vec beta0(10);
  arma::rowvec Ygamma(m);
  arma::rowvec gammanew(q);
  //arma::cube L(max(Ds),q,S);
  //arma::cube P(max(Ds),q,S);
  int lens, ns, nt;
  double fnow, fnew, likelinow, likelinew, xbeta, va;
  for(int i=0; i<S; i++){
    //cout<<i<<endl;
    lens=getind(E,i+1).n_cols;
    inds.set_size(1,lens);
    inds = getind(E,i+1);
    Gammai.set_size(Ds(i),q);
    Gammai = getGammai(Gamma,i,Ds(i),q);
    C.set_size(Ds(i), Ds(i));
    C = updateC(Gammai,theta,tau);
    if(Ds(i)>1){
      Cminus.set_size(Ds(i)-1, Ds(i)-1);
      b.set_size(1,Ds(i)-1);
      temp2.set_size(1,Ds(i));
      bnew.set_size(1,Ds(i)-1);
    }
    for(int j=0; j<Ds(i); j++){
      //cout<<j<<endl;
      ns = getind(R.row(i),j+1).n_cols;
      indi.set_size(1,ns);
      indi = getind(R.row(i),j+1);
      beta0.set_size(ns,1);
      temp.set_size(1,ns);
      for(int k = 0; k<ns; k++){
        //k in R is indi(k) now
        temp(k) =  findT(D.col(indi(k)).t());
      }
      nt = Nuni(temp);
      indT.set_size(1,nt);
      indT = unique(temp);
      if(Ds(i)==1){
        //need to change is we change the kernel function
        fnow = 1;
      }
      else{
        Cminus = removeii(C,j+1,j+1);
        b = removei(C.row(j),j+1);
        temp1 = b*inv(Cminus)*b.t();
        fnow = kernelC(Gammai.row(j),Gammai.row(j),theta,tau) - temp1(0,0);
      }
      likelinow = 0;
      for(int k = 0; k<lens;k++){
        //k in R is now inds(k)
        xbeta = sum(X.row(inds(k))%Beta.row(i));
        Ygamma = Y.row(inds(k)) -  xbeta;
        beta0 = rowsome(Z,indi)* Gammai.row(j).t();
        likelinow += dmvnrm_arma(colsome(Ygamma,indi), beta0.t(), sigma, true);
        for(int t=0; t<nt; t++){
          //t in R is now indT(t)
          va = min(5.0, mustar(inds(k),indT(t)));
          va = max(va, -5.0);
          likelinow += R::pnorm(va,0,1,true, true)*delta(inds(k),indT(t)) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(inds(k),indT(t)));
        }
      }
      //update Gamma one element by one element
      for(int l=0; l<q; l++){
        gammanew = Gammai.row(j);
        gammanew(l) = gammanew(l) + step(j,l,i);
        likelinew = 0;
        for(int k=0; k<lens; k++){
          //k in R is now inds(k)
          xbeta = sum(X.row(inds(k))%Beta.row(i));
          Ygamma = Y.row(inds(k)) -  xbeta;
          beta0 = rowsome(Z,indi)* gammanew.t();
          likelinew += dmvnrm_arma(colsome(Ygamma,indi), beta0.t(), sigma, true);
          munew = putvalue(munew,inds(k),indi,
                           computemu(mu,inds(k),indi, rowsome(Z,indi), Gammai.row(j), gammanew));
          mustarnew = putvalue(mustarnew,inds(k),indT,
                               computemustar(mustar, inds(k),indT, c, D, mu.row(inds(k)), munew.row(inds(k)), T0));
          for(int t=0; t<nt; t++){
            //t in R is now indT(t)
            va = min(5.0, mustarnew(inds(k),indT(t)));
            va = max(va, -5.0);
            likelinew += R::pnorm(va,0,1,true, true)*delta(inds(k),indT(t)) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(inds(k),indT(t)));
            
          }
          
        }
        if(Ds(i)==1){
          fnew = 1;
          //need to change is we change the kernel function
        }
        else{
          for(int k=0; k<Ds(i);k++){
            temp2(k) = kernelC(Gammai.row(k), gammanew,theta,tau);
          }
          bnew = removei(temp2, j+1);
          temp1 = bnew*inv(Cminus)*bnew.t();
          fnew = kernelC(gammanew,gammanew,theta, tau) - temp1(0,0);
        }
        //L(j,l,i) = likelinew - likelinow;
        //P(j,l,i) = log(fnew/fnow);
        if((lens==0) || (ns==0)){
          likelinew = 0;
          likelinow = 0;
        }
        if((log(fnew/fnow) + likelinew - likelinow)>log(runif(j,l,i))){
          if(Ds(i)>1){
            fnow = fnew;
            C.row(j) = temp2;
            C.col(j) = temp2.t();
            C(j,j) = kernelC(gammanew,gammanew,theta,tau);
          }
          mu = munew;
          mustar = mustarnew;
          likelinow = likelinew;
          Gammai.row(j) = gammanew;
        }
        else{
          munew = mu;
          mustarnew = mustar;
        }
      }
    }
    //Gamma[,,i] = Gammai
    Gamma = putGamma(Gamma, Gammai, i, q);
  }
  ans["mu"]= mu;
  ans["mustar"] = mustar;
  ans["Gamma"]= Gamma;
  //ans["L"] = L;
  //ans["P"] = P;
  return(ans);
}


// [[Rcpp::export]]
List updateGammanomiss(arma::mat X,
                       arma::mat Y,
                       arma::mat Z,
                       arma::mat delta,
                       arma::mat Beta,
                       arma::cube Gamma,
                       arma::rowvec E,
                       arma::mat R,
                       double S,
                       arma::rowvec Ds,
                       arma::mat mu,
                       double sigma,
                       arma::cube step,
                       arma::cube runif,
                       int n, int m, int T0, int p, int q, arma::mat D,
                       double theta, double tau){
  List ans;
  arma::mat mustarnew(n, T0);
  arma::mat munew(n, m);
  arma::rowvec inds(n);
  arma::rowvec indi(m);
  arma::rowvec indT(T0);
  arma::mat Gammai(max(Ds),q);
  arma::rowvec temp(T0);
  munew =mu;
  arma::mat C(10,10);
  arma::mat Cminus(9,9);
  arma::mat temp1(1,1);
  arma::rowvec b(9);
  arma::rowvec temp2(10);
  arma::rowvec bnew(9);
  arma::vec beta0(10);
  arma::rowvec Ygamma(m);
  arma::rowvec gammanew(q);
  //arma::cube L(max(Ds),q,S);
  //arma::cube P(max(Ds),q,S);
  int lens, ns, nt;
  double fnow, fnew, likelinow, likelinew, xbeta;
  for(int i=0; i<S; i++){
    //cout<<i<<endl;
    lens=getind(E,i+1).n_cols;
    inds.set_size(1,lens);
    inds = getind(E,i+1);
    Gammai.set_size(Ds(i),q);
    Gammai = getGammai(Gamma,i,Ds(i),q);
    C.set_size(Ds(i), Ds(i));
    C = updateC(Gammai,theta,tau);
    if(Ds(i)>1){
      Cminus.set_size(Ds(i)-1, Ds(i)-1);
      b.set_size(1,Ds(i)-1);
      temp2.set_size(1,Ds(i));
      bnew.set_size(1,Ds(i)-1);
    }
    for(int j=0; j<Ds(i); j++){
      //cout<<j<<endl;
      ns = getind(R.row(i),j+1).n_cols;
      indi.set_size(1,ns);
      indi = getind(R.row(i),j+1);
      beta0.set_size(ns,1);
      temp.set_size(1,ns);
      for(int k = 0; k<ns; k++){
        //k in R is indi(k) now
        temp(k) =  findT(D.col(indi(k)).t());
      }
      nt = Nuni(temp);
      indT.set_size(1,nt);
      indT = unique(temp);
      if(Ds(i)==1){
        //need to change is we change the kernel function
        fnow = 1;
      }
      else{
        Cminus = removeii(C,j+1,j+1);
        b = removei(C.row(j),j+1);
        temp1 = b*inv(Cminus)*b.t();
        fnow = kernelC(Gammai.row(j),Gammai.row(j),theta,tau) - temp1(0,0);
      }
      likelinow = 0;
      for(int k = 0; k<lens;k++){
        //k in R is now inds(k)
        xbeta = sum(X.row(inds(k))%Beta.row(i));
        Ygamma = Y.row(inds(k)) -  xbeta;
        beta0 = rowsome(Z,indi)* Gammai.row(j).t();
        likelinow += dmvnrm_arma(colsome(Ygamma,indi), beta0.t(), sigma, true);
        
      }
      //update Gamma one element by one element
      for(int l=0; l<q; l++){
        gammanew = Gammai.row(j);
        gammanew(l) = gammanew(l) + step(j,l,i);
        likelinew = 0;
        for(int k=0; k<lens; k++){
          //k in R is now inds(k)
          xbeta = sum(X.row(inds(k))%Beta.row(i));
          Ygamma = Y.row(inds(k)) -  xbeta;
          beta0 = rowsome(Z,indi)* gammanew.t();
          likelinew += dmvnrm_arma(colsome(Ygamma,indi), beta0.t(), sigma, true);
          munew = putvalue(munew,inds(k),indi,
                           computemu(mu,inds(k),indi, rowsome(Z,indi), Gammai.row(j), gammanew));
          
          
        }
        if(Ds(i)==1){
          fnew = 1;
          //need to change is we change the kernel function
        }
        else{
          for(int k=0; k<Ds(i);k++){
            temp2(k) = kernelC(Gammai.row(k), gammanew,theta,tau);
          }
          bnew = removei(temp2, j+1);
          temp1 = bnew*inv(Cminus)*bnew.t();
          fnew = kernelC(gammanew,gammanew,theta, tau) - temp1(0,0);
        }
        //L(j,l,i) = likelinew - likelinow;
        //P(j,l,i) = log(fnew/fnow);
        if((lens==0) || (ns==0)){
          likelinew = 0;
          likelinow = 0;
        }
        if((log(fnew/fnow) + likelinew - likelinow)>log(runif(j,l,i))){
          if(Ds(i)>1){
            fnow = fnew;
            C.row(j) = temp2;
            C.col(j) = temp2.t();
            C(j,j) = kernelC(gammanew,gammanew,theta,tau);
          }
          mu = munew;
          likelinow = likelinew;
          Gammai.row(j) = gammanew;
        }
        else{
          munew = mu;
        }
      }
    }
    //Gamma[,,i] = Gammai
    Gamma = putGamma(Gamma, Gammai, i, q);
  }
  ans["mu"]= mu;
  ans["Gamma"]= Gamma;
  //ans["L"] = L;
  //ans["P"] = P;
  return(ans);
}

// [[Rcpp::export]]
List updateE(arma::mat Beta,
             arma::cube Gamma,
             arma::rowvec w,
             arma::mat X,
             arma::mat Y,
             arma::mat Z,
             arma::mat delta,
             arma::rowvec E,
             arma::mat R,
             double S,
             arma::rowvec Ds,
             arma::mat mu,
             arma::mat mustar,
             double sigma,
             arma::rowvec c,
             int n, int m, int T0, int p, int q, arma::mat D){
  List ans;
  arma::rowvec pro(S);
  arma::vec beta0(m);
  arma::rowvec gammaij(q);
  arma::rowvec indj(m);
  arma::vec mustarnew(T0);
  int len;
  arma::mat Zj(m,q);
  double likeli, va;
  arma::vec temp3(m);
  for(int i=0; i<n;i++){
    //compute the likeli
    pro = w;
    for(int k=0; k<S; k++){
      beta0 = matrix(X.row(i),m)*Beta.row(k).t() ;
      
      for(int j=0; j<Ds(k); j++){
        len=getind(R.row(k),j+1).n_cols;
        indj.set_size(1,len);
        if(len == 0){
          continue;
        }
        indj = getind(R.row(k),j+1);
        Zj.set_size(len,q);
        Zj = rowsome(Z,indj);
        gammaij = getgamma(Gamma, k,j);
        temp3.set_size(len,1);
        temp3 = Zj*gammaij.t();
        beta0 = minusvalue(beta0.t(), 0, indj, -temp3.t()).t();
      }
      
      
      
      
      likeli = dmvnrm_arma(Y.row(i),beta0.t(),sigma, true);
      mustarnew = mustar.row(i).t() -cDmu(c,D*mu.row(i).t()) + cDmu(c, D*beta0);
      for(int t=0; t<T0; t++){
        va = min(5.0, mustarnew(t));
        va = max(va, -5.0);
        likeli += R::pnorm(va,0,1,true,true)*delta(i,t) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(i,t));
      }
      pro(k) = log(pro(k)) + likeli;
    }
    
    
    pro = pro -max(pro);
    pro = exp(pro);
    pro = pro/sum(pro);
    //assign indicator
    E(i) =  rmunoim(pro);
    //update mu and mustar
    beta0 = matrix(X.row(i),m)*Beta.row(E(i)-1).t() ;
    for(int j=0; j<Ds(E(i)-1); j++){
      len=getind(R.row(E(i)-1),j+1).n_cols;
      indj.set_size(1,len);
      indj = getind(R.row(E(i)-1),j+1);
      Zj.set_size(len,q);
      Zj = rowsome(Z,indj);
      gammaij = getgamma(Gamma, E(i)-1,j);
      temp3.set_size(len,1);
      temp3 = Zj*gammaij.t();
      beta0 = minusvalue(beta0.t(), 0, indj, -temp3.t()).t();
    }
    mustarnew = mustar.row(i).t() -cDmu(c,D*mu.row(i).t()) + cDmu(c, D*beta0);
    mu.row(i) = beta0.t();
    mustar.row(i) = mustarnew.t();
  }
  ans["E"] = E;
  ans["Ds"] = Ds;
  ans["mu"] = mu;
  ans["mustar"] = mustar;
  return(ans);
}


// [[Rcpp::export]]
List updateEnomiss(arma::mat Beta,
                   arma::cube Gamma,
                   arma::rowvec w,
                   arma::mat X,
                   arma::mat Y,
                   arma::mat Z,
                   arma::mat delta,
                   arma::rowvec E,
                   arma::mat R,
                   double S,
                   arma::rowvec Ds,
                   arma::mat mu,
                   double sigma,
                   int n, int m, int T0, int p, int q, arma::mat D){
  List ans;
  arma::rowvec pro(S);
  arma::vec beta0(m);
  arma::rowvec gammaij(q);
  arma::rowvec indj(m);
  int len;
  arma::mat Zj(m,q);
  double likeli;
  arma::vec temp3(m);
  for(int i=0; i<n;i++){
    //compute the likeli
    pro = w;
    for(int k=0; k<S; k++){
      beta0 = matrix(X.row(i),m)*Beta.row(k).t() ;
      
      for(int j=0; j<Ds(k); j++){
        len=getind(R.row(k),j+1).n_cols;
        indj.set_size(1,len);
        if(len == 0){
          continue;
        }
        indj = getind(R.row(k),j+1);
        Zj.set_size(len,q);
        Zj = rowsome(Z,indj);
        gammaij = getgamma(Gamma, k,j);
        temp3.set_size(len,1);
        temp3 = Zj*gammaij.t();
        beta0 = minusvalue(beta0.t(), 0, indj, -temp3.t()).t();
      }
      
      likeli = dmvnrm_arma(Y.row(i),beta0.t(),sigma, true);
      
      pro(k) = log(pro(k)) + likeli;
    }
    
    
    pro = pro -max(pro);
    pro = exp(pro);
    pro = pro/sum(pro);
    //assign indicator
    E(i) =  rmunoim(pro);
    //update mu and mustar
    beta0 = matrix(X.row(i),m)*Beta.row(E(i)-1).t() ;
    for(int j=0; j<Ds(E(i)-1); j++){
      len=getind(R.row(E(i)-1),j+1).n_cols;
      indj.set_size(1,len);
      indj = getind(R.row(E(i)-1),j+1);
      Zj.set_size(len,q);
      Zj = rowsome(Z,indj);
      gammaij = getgamma(Gamma, E(i)-1,j);
      temp3.set_size(len,1);
      temp3 = Zj*gammaij.t();
      beta0 = minusvalue(beta0.t(), 0, indj, -temp3.t()).t();
    }
    mu.row(i) = beta0.t();
  }
  ans["E"] = E;
  ans["Ds"] = Ds;
  ans["mu"] = mu;
  return(ans);
}

// [[Rcpp::export]]
arma::mat updatemu(arma::mat R,
                   arma::mat Z,
                   arma::mat X,
                   arma::cube Gamma,
                   arma::rowvec K,
                   arma::mat Beta,
                   arma::rowvec E,
                   int m, int n,int p, int q){
  arma::mat mu(n,m);
  arma::rowvec Xi(p);
  arma::rowvec Betai(p);
  arma::rowvec gammaij(q);
  arma::mat mui_temp(1,1);
  arma::mat Zj = Z;
  arma::rowvec indj(m);
  arma::vec temp1(m);
  int len;
  for(int i=0; i<n; i++){
    Xi = X.row(i);
    Betai = Beta.row(E(i)-1);
    mui_temp = sum(Xi%Betai);
    mu.row(i) = arma::repmat(mui_temp,1,m);
    for(int j=0; j<K(E(i)-1); j++){
      gammaij = Gamma.slice(E(i)-1).row(j);
      len = getind(R.row(E(i)-1),j+1).n_cols;
      indj.set_size(1,len);
      indj = getind(R.row(E(i)-1),j+1);
      Zj.set_size(len,q);
      Zj=rowsome(Z,indj);
      temp1.set_size(len,1);
      temp1 = Zj*gammaij.t();
      mu = minusvalue(mu, i, indj, -temp1.t());
    }
  }
  return (mu);
}

arma::rowvec putvalue_vec(arma::rowvec x, arma::rowvec indi, arma::colvec value){
  int n = indi.n_cols;
  for(int i=0; i<n; i++){
    x(indi(i)) = value(i);
  }
  return (x);
}

arma::mat rbind(arma::mat x, arma::rowvec y, arma::rowvec z){
  int R = x.n_rows+2;
  int C = x.n_cols;
  arma::mat ans(R,C);
  for(int i=0; i<R-2; i++){
    ans.row(i) = x.row(i);
  }
  ans.row(R-2) = y;
  ans.row(R-1) = z;
  return (ans);
}

arma::mat rbind0(arma::mat x, arma::rowvec y){
  int R = x.n_rows+1;
  int C = x.n_cols;
  arma::mat ans(R,C);
  for(int i=0; i<R-1; i++){
    ans.row(i) = x.row(i);
  }
  ans.row(R-1) = y;
  return (ans);
}

arma::rowvec connect(arma::rowvec x, double y, double z){
  int n = x.n_cols;
  x.reshape(1,n+2);
  x(n) = y;
  x(n+1) = z;
  return(x);
}

arma::rowvec connect0(arma::rowvec x, double y){
  int n = x.n_cols;
  x.reshape(1,n+1);
  x(n) = y;
  return(x);
}


// [[Rcpp::export]]
List Split(arma::rowvec w,
           int K,
           arma::mat Gamma,
           arma::rowvec Beta,
           arma::mat X,
           arma::mat Y,
           arma::mat Z,
           arma::rowvec R,
           arma::mat delta,
           arma::rowvec mu,
           arma::mat mu_star,
           double c,
           double sigma_square,
           arma::mat C,
           double theta, double tau,
           int m, int n, int q, int T0, double hyper_delta = 1){
  List ans;
  int Knew = K + 1;
  int l = sampleint(K);
  double w_1 = w(l-1);
  int nid, Tl;
  double va;
  arma::mat temp(1,1);
  //this temp is used for repmat
  arma::rowvec gamma_1(q);
  nid=getind(R,l).n_cols;
  arma::rowvec l_label(nid);
  arma::rowvec indT(nid);
  l_label = getind(R,l);
  for(int i=0; i<nid; i++){
    indT(i) = ceil((l_label(i) + 1)/6.0) - 1;
  }
  int nt = Nuni(indT);
  arma::rowvec indT_unique(nt);
  indT_unique = unique(indT);
  arma::mat Y_use(n,nid);
  arma::mat Z_use(nid,m);
  Z_use = rowsome(Z,l_label);
  Y_use = colsome(Y,l_label);
  
  //re-arrange labels
  arma::rowvec R_new(m);
  
  for(int i=0; i<m; i++){
    if(R(i)>l){
      R_new(i) = R(i)-1;
    }
    else{
      R_new(i) = R(i);
    }
  }
  //compute the split transformation using moment matching principle
  arma::rowvec beta(q);
  for(int i=0; i<q; i++){
    beta(i) = R::rbeta(2,2);
  }
  double alpha = R::runif(0,1);
  double w_new_1 = w_1 * alpha;
  double w_new_2 = w_1 * (1-alpha);
  arma::rowvec pro(2);
  pro(0) = log(w_new_1);
  pro(1) = log(w_new_2);
  arma::rowvec gamma_new_1(q);
  arma::rowvec gamma_new_2(q);
  double ratio_propose = 0;
  for(int i=0; i<q; i++){
    gamma_new_1(i) = gamma_1(i) - std::sqrt((1-alpha)/alpha) * beta(i);
    gamma_new_2(i) = gamma_1(i) + std::sqrt(alpha/(1-alpha)) * beta(i);
    ratio_propose += R::dbeta(beta(i),2,2,true);
  }
  
  //compute the proposal ratio
  arma::rowvec split_to_two(nid);
  split_to_two.fill(0);
  int n1 = 0;
  int n2 = 0;
  double ratio_like = 0;
  if(nid>0){
    for(int i=0; i<nid; i++){
      temp(0,0) = sum(gamma_new_1 % Z_use.row(i));
      pro(0) += dmvnrm_arma(Y_use.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      temp(0,0) = sum(gamma_new_2 % Z_use.row(i));
      pro(1) += dmvnrm_arma(Y_use.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      pro = pro -max(pro);
      pro = exp(pro);
      pro = pro/sum(pro);
      if(R::runif(0,1)<pro(0)){
        split_to_two(i) = 1;
        ratio_propose += log(pro(0));
        n1 += 1;
        R_new(l_label(i)) = K;
      }
      else{
        ratio_propose += log(pro(1));
        n2 += 1;
        R_new(l_label(i)) = Knew;
      }
      
    }
    
    arma::rowvec ind1(n1);
    arma::rowvec ind2(n2);
    ind1 = getind(split_to_two, 1);
    ind2 = getind(split_to_two, 0);
    arma::rowvec mu_new(nid);
    arma::rowvec mu_now(nid);
    mu_now = colsome(mu, l_label);
    mu_new = mu_now;
    double likeli_now = 0;
    double likeli_new = 0;
    arma::mat Z_1(n1,q);
    arma::mat Z_2(n2,q);
    Z_1 = rowsome(Z_use, ind1);
    Z_2 = rowsome(Z_use, ind2);
    mu_new = putvalue_vec(mu_new, ind1, Z_1*gamma_new_1.t());
    mu_new = putvalue_vec(mu_new, ind2, Z_2*gamma_new_2.t());
    for(int i=0; i<n; i++){
      likeli_now += dmvnrm_arma(Y_use.row(i), mu_now,  sigma_square, true);
      likeli_new += dmvnrm_arma(Y_use.row(i), mu_new,  sigma_square, true);
    }
    
    arma::mat mu_star_now(n,T0);
    arma::mat mu_star_new(n,T0);
    mu_star_new = mu_star;
    mu_star_now = mu_star;
    
    for(int j=0; j<n; j++){
      for(int i=0; i<nid; i++){
        mu_star_new(j,indT(i)) = mu_star_new(j,indT(i)) - c * mu_now(i) * (1.0/6) + c * mu_new(i) * (1.0/6);
      }
    }
    
    for(int j=0; j<n; j++){
      for(int i=0; i<nt; i++){
        Tl = indT_unique(i);
        va = min(5.0, mu_star_now(j,Tl) );
        va = max(va, -5.0);
        likeli_now += R::pnorm(va,0,1,true,true)*delta(j,Tl) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(j,Tl));
        va = min(5.0, mu_star_new(j,Tl) );
        va = max(va, -5.0);
        likeli_new += R::pnorm(va,0,1,true,true)*delta(j,Tl) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(j,Tl));
      }
    }
    
    ratio_like = likeli_new - likeli_now;
  }
  
  //compute the log Jacobian
  double log_Jacobian = log(w_1) - (q/2.0) * log(alpha*(1-alpha));
  
  //compute the likelihood ratio
  
  //compute the prior ratio
  double ratio_prior_w = (hyper_delta - 1 + n1) * log(w_new_1) + (hyper_delta - 1 + n2) * log(w_new_2) - 
    (hyper_delta - 1 + nid) * log(w_1) - log(R::beta(hyper_delta, K * hyper_delta));
  
  arma::mat Gamma_new(Knew,q);
  arma::mat C_new(Knew,Knew);
  arma::rowvec w_new(Knew);
  if(n1==0){
    Knew -= 1;
    Gamma_new.set_size(Knew,q);
    C_new.set_size(Knew,Knew);
    w_new.set_size(1,Knew);
    w_new = connect0(removei(w,l),w_new_2);
    Gamma_new = rbind0(removeiii(Gamma, l),gamma_new_2);
    for(int i=0; i<nid; i++){
      R_new(l_label(i)) -= 1;
    }
    
  }else if(n2==0){
    Knew -= 1;
    Gamma_new.set_size(Knew,q);
    C_new.set_size(Knew,Knew);
    w_new.set_size(1,Knew);
    w_new = connect0(removei(w,l),w_new_1);
    Gamma_new = rbind0(removeiii(Gamma, l),gamma_new_1);
    
  }else{
    w_new = connect(removei(w,l), w_new_1, w_new_2);
    Gamma_new = rbind(removeiii(Gamma, l),gamma_new_1, gamma_new_2);
  }
  C_new = updateC(Gamma_new,theta,tau);
  double ratio_prior_gamma = log(det(C_new)) - log(det(C));
  double ratio_prior = ratio_prior_gamma + ratio_prior_w;
  double birth_ratio = ratio_like + ratio_prior - log(Knew) + log_Jacobian - log(Knew) - ratio_propose;
  if(Knew == 2){
    birth_ratio -= log(2); 
  }
  if(log(R::runif(0,1)) < birth_ratio){
    K=Knew;
    Gamma = Gamma_new;
    R = R_new;
    C = C_new;
    w = w_new;
  }
  
  ans["K"] = K;
  ans["w"] = w;
  ans["Gamma"] = Gamma;
  ans["R"]= R;
  ans["C"] = C;
  return (ans);
}


// [[Rcpp::export]]
List Splitnomiss(arma::rowvec w,
                 int K,
                 arma::mat Gamma,
                 arma::rowvec Beta,
                 arma::mat X,
                 arma::mat Y,
                 arma::mat Z,
                 arma::rowvec R,
                 arma::mat delta,
                 arma::rowvec mu,
                 double sigma_square,
                 arma::mat C,
                 double theta, double tau,
                 int m, int n, int q, int T0, double hyper_delta = 1){
  List ans;
  int Knew = K + 1;
  int l = sampleint(K);
  double w_1 = w(l-1);
  int nid;
  arma::mat temp(1,1);
  //this temp is used for repmat
  arma::rowvec gamma_1(q);
  nid=getind(R,l).n_cols;
  arma::rowvec l_label(nid);
  arma::rowvec indT(nid);
  l_label = getind(R,l);
  for(int i=0; i<nid; i++){
    indT(i) = ceil((l_label(i) + 1)/6.0) - 1;
  }
  int nt = Nuni(indT);
  arma::rowvec indT_unique(nt);
  indT_unique = unique(indT);
  arma::mat Y_use(n,nid);
  arma::mat Z_use(nid,m);
  Z_use = rowsome(Z,l_label);
  Y_use = colsome(Y,l_label);
  
  //re-arrange labels
  arma::rowvec R_new(m);
  
  for(int i=0; i<m; i++){
    if(R(i)>l){
      R_new(i) = R(i)-1;
    }
    else{
      R_new(i) = R(i);
    }
  }
  //compute the split transformation using moment matching principle
  arma::rowvec beta(q);
  for(int i=0; i<q; i++){
    beta(i) = R::rbeta(2,2);
  }
  double alpha = R::runif(0,1);
  double w_new_1 = w_1 * alpha;
  double w_new_2 = w_1 * (1-alpha);
  arma::rowvec pro(2);
  pro(0) = log(w_new_1);
  pro(1) = log(w_new_2);
  arma::rowvec gamma_new_1(q);
  arma::rowvec gamma_new_2(q);
  double ratio_propose = 0;
  for(int i=0; i<q; i++){
    gamma_new_1(i) = gamma_1(i) - std::sqrt((1-alpha)/alpha) * beta(i);
    gamma_new_2(i) = gamma_1(i) + std::sqrt(alpha/(1-alpha)) * beta(i);
    ratio_propose += R::dbeta(beta(i),2,2,true);
  }
  
  //compute the proposal ratio
  arma::rowvec split_to_two(nid);
  split_to_two.fill(0);
  int n1 = 0;
  int n2 = 0;
  double ratio_like = 0;
  if(nid>0){
    for(int i=0; i<nid; i++){
      temp(0,0) = sum(gamma_new_1 % Z_use.row(i));
      pro(0) += dmvnrm_arma(Y_use.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      temp(0,0) = sum(gamma_new_2 % Z_use.row(i));
      pro(1) += dmvnrm_arma(Y_use.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      pro = pro -max(pro);
      pro = exp(pro);
      pro = pro/sum(pro);
      if(R::runif(0,1)<pro(0)){
        split_to_two(i) = 1;
        ratio_propose += log(pro(0));
        n1 += 1;
        R_new(l_label(i)) = K;
      }
      else{
        ratio_propose += log(pro(1));
        n2 += 1;
        R_new(l_label(i)) = Knew;
      }
      
    }
    
    arma::rowvec ind1(n1);
    arma::rowvec ind2(n2);
    ind1 = getind(split_to_two, 1);
    ind2 = getind(split_to_two, 0);
    arma::rowvec mu_new(nid);
    arma::rowvec mu_now(nid);
    mu_now = colsome(mu, l_label);
    mu_new = mu_now;
    double likeli_now = 0;
    double likeli_new = 0;
    arma::mat Z_1(n1,q);
    arma::mat Z_2(n2,q);
    Z_1 = rowsome(Z_use, ind1);
    Z_2 = rowsome(Z_use, ind2);
    mu_new = putvalue_vec(mu_new, ind1, Z_1*gamma_new_1.t());
    mu_new = putvalue_vec(mu_new, ind2, Z_2*gamma_new_2.t());
    for(int i=0; i<n; i++){
      likeli_now += dmvnrm_arma(Y_use.row(i), mu_now,  sigma_square, true);
      likeli_new += dmvnrm_arma(Y_use.row(i), mu_new,  sigma_square, true);
    }
    
    
    
    ratio_like = likeli_new - likeli_now;
  }
  
  //compute the log Jacobian
  double log_Jacobian = log(w_1) - (q/2.0) * log(alpha*(1-alpha));
  
  //compute the likelihood ratio
  
  //compute the prior ratio
  double ratio_prior_w = (hyper_delta - 1 + n1) * log(w_new_1) + (hyper_delta - 1 + n2) * log(w_new_2) - 
    (hyper_delta - 1 + nid) * log(w_1) - log(R::beta(hyper_delta, K * hyper_delta));
  
  arma::mat Gamma_new(Knew,q);
  arma::mat C_new(Knew,Knew);
  arma::rowvec w_new(Knew);
  if(n1==0){
    Knew -= 1;
    Gamma_new.set_size(Knew,q);
    C_new.set_size(Knew,Knew);
    w_new.set_size(1,Knew);
    w_new = connect0(removei(w,l),w_new_2);
    Gamma_new = rbind0(removeiii(Gamma, l),gamma_new_2);
    for(int i=0; i<nid; i++){
      R_new(l_label(i)) -= 1;
    }
    
  }else if(n2==0){
    Knew -= 1;
    Gamma_new.set_size(Knew,q);
    C_new.set_size(Knew,Knew);
    w_new.set_size(1,Knew);
    w_new = connect0(removei(w,l),w_new_1);
    Gamma_new = rbind0(removeiii(Gamma, l),gamma_new_1);
    
  }else{
    w_new = connect(removei(w,l), w_new_1, w_new_2);
    Gamma_new = rbind(removeiii(Gamma, l),gamma_new_1, gamma_new_2);
  }
  C_new = updateC(Gamma_new,theta,tau);
  double ratio_prior_gamma = log(det(C_new)) - log(det(C));
  double ratio_prior = ratio_prior_gamma + ratio_prior_w;
  double birth_ratio = ratio_like + ratio_prior - log(Knew) + log_Jacobian - log(Knew) - ratio_propose;
  if(Knew == 2){
    birth_ratio -= log(2); 
  }
  if(log(R::runif(0,1)) < birth_ratio){
    K=Knew;
    Gamma = Gamma_new;
    R = R_new;
    C = C_new;
    w = w_new;
  }
  
  ans["K"] = K;
  ans["w"] = w;
  ans["Gamma"] = Gamma;
  ans["R"]= R;
  ans["C"] = C;
  return (ans);
}


arma::rowvec sampleint2(int K){
  arma::rowvec ans(2);
  int a = sampleint(K);
  int b = sampleint(K-1);
  if(b>=a){
    b += 1;
    ans(0) = a;
    ans(1) =b;
  }
  else{
    ans(0)=b;
    ans(1) = a;
  }
  return (ans);
}

// [[Rcpp::export]]
List Merge(arma::rowvec w,
           int K,
           arma::mat Gamma,
           arma::rowvec Beta,
           arma::mat X,
           arma::mat Y,
           arma::mat Z,
           arma::rowvec R,
           arma::mat delta,
           arma::rowvec mu,
           arma::mat mu_star,
           double c,
           double sigma_square,
           arma::mat C,
           double theta, double tau,
           int m, int n, int q, int T0, double hyper_delta = 1){
  List ans;
  int Knew = K-1;
  
  //Uniformly pick two labels that are potentially to be merged
  arma::rowvec id(2);
  id = sampleint2(K);
  int id1 = id(0);
  int id2 = id(1);
  double w1 = w(id1-1);
  double w2 = w(id2-1);
  arma::rowvec pro_init(2);
  arma::rowvec pro(2);
  pro_init(0) = log(w1);
  pro_init(1) = log(w2);
  arma::rowvec gamma_1(q);
  arma::rowvec gamma_2(q);
  arma::rowvec gamma_new(q);
  gamma_1 = Gamma.row(id1-1);
  gamma_2 = Gamma.row(id2-1);
  int Tl;
  double va;
  // Find those groups that are associated with the kth mixture
  int n1=getind(R,id1).n_cols;
  int n2=getind(R,id2).n_cols;
  arma::mat Z1(n,q);
  arma::mat Y1(n,n);
  arma::mat Z2(n,q);
  arma::mat Y2(n,n);
  arma::rowvec ind1(m);
  arma::rowvec ind2(m);
  int nid = n1+n2;
  if(n1>0){
    ind1.set_size(n1);
    ind1 = getind(R,id1);
    Z1.set_size(n1,q);
    Y1.set_size(n,n1);
    Z1 = rowsome( Z, ind1);
    Y1 = colsome(Y, ind1);
  }
  if(n2>0){
    ind2.set_size(n2);
    ind2 = getind(R,id2);
    Z2.set_size(n2,q);
    Y2.set_size(n,n2);
    Z2 = rowsome(Z,ind2);
    Y2 = colsome(Y, ind2);
  }
  
  //re-arrange labels
  arma::rowvec R_new(m);
  R_new = R;
  for(int i=0; i<m; i++){
    if((R(i)>id1) & (R(i)<id2)){
      R_new(i)= R(i) -1;
    }
    if((R(i)>id2)){
      R_new(i)= R(i) -2;
    }
    if((R(i)==id1) || (R(i)==id2)){
      R_new(i) = Knew;
    }
  }
  
  //Compute the merge transformation using moment-matching principle
  double w1_new = w1 + w2;
  arma::rowvec w_new(Knew);
  w_new(Knew - 1) = w1_new;
  for(int i=0; i<Knew-1; i++){
    if(i<id1-1){
      w_new(i) = w(i);
    }
    if((i>=id1-1) & (i<id2-1)){
      w_new(i) = w(i+1);
    }
    if(i>=id2-1){
      w_new(i) = w(i+2);
    }
  }
  
  double alpha = w1/w1_new;
  arma::mat Gamma_new(Knew,q);
  gamma_new = (w1 * gamma_1 + w2 * gamma_2)/w1_new;
  Gamma_new = rbind0(removeiii(removeiii(Gamma,id1), id2-1),gamma_new);
  
  arma::rowvec beta(q);
  double ratio_propose = 0;
  for(int i=0; i<q; i++){
    beta(i) = (gamma_new(i) - gamma_1(i)) * std::sqrt(alpha/(1-alpha));
    if((beta(i)<1) & (beta(i)>0)){
      ratio_propose += R::dbeta(beta(i),2,2,true);
    }
  }
  //compute the proposal ratio
  double diff = 0;
  arma::mat temp(1,1);
  if(n1>0){
    for(int i=0; i<n1; i++){
      pro = pro_init;
      temp(0,0) = sum(gamma_1 % Z1.row(i));
      pro(0) += dmvnrm_arma(Y1.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      temp(0,0) = sum(gamma_2 % Z1.row(i));
      pro = pro -max(pro);
      pro = exp(pro);
      pro = pro/sum(pro);
      diff = pro(1) - pro(0);
      pro(0) += 0.00001*diff;
      pro(1) -= 0.00001*diff;
      ratio_propose += log(pro(0));
    }
  }
  if(n2>0){
    for(int i=0; i<n2; i++){
      pro = pro_init;
      temp(0,0) = sum(gamma_1 % Z2.row(i));
      pro(0) += dmvnrm_arma(Y2.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      temp(0,0) = sum(gamma_2 % Z2.row(i));
      pro(1) += dmvnrm_arma(Y2.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      pro = pro -max(pro);
      pro = exp(pro);
      pro = pro/sum(pro);
      diff = pro(1) - pro(0);
      pro(0) += 0.00001*diff;
      pro(1) -= 0.00001*diff;
      ratio_propose += log(pro(1));
    }
  }
  
  //compute the log Jacobian of moment-matching transformation
  double log_Jacobian = log(w1_new) - (q/2.0) * log(alpha*(1-alpha));
  //compute the likelihood ratio
  double likeli_now = 0;
  double likeli_new = 0;
  arma::mat mu_star_now(n,T0);
  arma::mat mu_star_new(n,T0);
  mu_star_new = mu_star;
  mu_star_now = mu_star;
  if(n1>0){
    arma::rowvec mu_new1(n1);
    arma::rowvec mu_now1(n1);
    arma::rowvec indT1(n1);
    mu_now1 = colsome(mu,ind1);
    mu_new1 = gamma_new * Z1.t();
    for(int i=0; i<n1; i++){
      indT1(i) = ceil((ind1(i) + 1)/6.0) - 1;
    }
    for(int i=0; i<n; i++){
      likeli_now += dmvnrm_arma(Y1.row(i) ,mu_now1,  sigma_square,true);
      likeli_new += dmvnrm_arma(Y1.row(i) ,mu_new1,  sigma_square,true);
    }
    for(int j=0; j<n; j++){
      for(int i=0; i<n1; i++){
        mu_star_new(j,indT1(i)) = mu_star_new(j,indT1(i)) - c * mu_now1(i) * (1.0/6) + c * mu_new1(i) * (1.0/6);
      }
    }
    
  }
  if(n2>0){
    arma::rowvec mu_new2(n2);
    arma::rowvec mu_now2(n2);
    arma::rowvec indT2(n2);
    mu_now2 = colsome(mu,ind2);
    mu_new2 = gamma_new * Z2.t();
    for(int i=0; i<n2; i++){
      indT2(i) = ceil((ind2(i) + 1)/6.0) - 1;
    }
    for(int i=0; i<n; i++){
      likeli_now += dmvnrm_arma(Y2.row(i) ,mu_now2, sigma_square,true);
      likeli_new += dmvnrm_arma(Y2.row(i) ,mu_new2, sigma_square,true);
    }
    for(int j=0; j<n; j++){
      for(int i=0; i<n2; i++){
        mu_star_new(j,indT2(i)) = mu_star_new(j,indT2(i)) - c * mu_now2(i) * (1.0/6) + c * mu_new2(i) * (1.0/6);
      }
    }
  }
  int flag = 0;
  arma::rowvec indT_temp(m);
  if(nid>0){
    //compute indT_unique
    for(int i=0; i<m; i++){
      if((R(i)==id1) || (R(i)==id2)){
        indT_temp(flag) = ceil((i+1)/6.0) - 1;
        flag += 1;
      }
    }
    indT_temp.reshape(1,flag);
    int nt = Nuni(indT_temp);
    arma::rowvec indT_unique(nt);
    indT_unique = unique(indT_temp);
    for(int j=0; j<n; j++){
      for(int i=0; i<nt; i++){
        Tl = indT_unique(i);
        va = min(5.0, mu_star_now(j,Tl) );
        va = max(va, -5.0);
        likeli_now += R::pnorm(va,0,1,true,true)*delta(j,Tl) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(j,Tl));
        va = min(5.0, mu_star_new(j,Tl) );
        va = max(va, -5.0);
        likeli_new += R::pnorm(va,0,1,true,true)*delta(j,Tl) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(j,Tl));
      }
    }
  }
  double ratio_like = likeli_new - likeli_now;
  
  //compute the prior ratio
  double ratio_prior_w = (hyper_delta - 1 + n1) * log(w1) + (hyper_delta - 1 + n2) * log(w2) - 
    (hyper_delta - 1 + nid) * log(w1_new) - log(R::beta(hyper_delta, Knew * hyper_delta));
  arma::mat C_new(Knew, Knew);
  C_new = updateC(Gamma_new,theta,tau);
  double ratio_prior_gamma = log(det(C)) - log(det(C_new));
  double ratio_prior = ratio_prior_w + ratio_prior_gamma;
  double death_ratio = log(K)*2 + ratio_propose + ratio_like - ratio_prior - log_Jacobian;
  if(Knew ==1){
    death_ratio = death_ratio + log(2);
  }
  
  if(log(R::runif(0,1)) < death_ratio){
    K=Knew;
    Gamma = Gamma_new;
    R = R_new;
    C = C_new;
    w = w_new;
  }
  
  ans["K"] = K;
  ans["w"] = w;
  ans["Gamma"] = Gamma;
  ans["R"]= R;
  ans["C"] = C;
  return (ans);
}

// [[Rcpp::export]]
List Mergenomiss(arma::rowvec w,
                 int K,
                 arma::mat Gamma,
                 arma::rowvec Beta,
                 arma::mat X,
                 arma::mat Y,
                 arma::mat Z,
                 arma::rowvec R,
                 arma::mat delta,
                 arma::rowvec mu,
                 double sigma_square,
                 arma::mat C,
                 double theta, double tau,
                 int m, int n, int q, int T0, double hyper_delta = 1){
  List ans;
  int Knew = K-1;
  
  //Uniformly pick two labels that are potentially to be merged
  arma::rowvec id(2);
  id = sampleint2(K);
  int id1 = id(0);
  int id2 = id(1);
  double w1 = w(id1-1);
  double w2 = w(id2-1);
  arma::rowvec pro_init(2);
  arma::rowvec pro(2);
  pro_init(0) = log(w1);
  pro_init(1) = log(w2);
  arma::rowvec gamma_1(q);
  arma::rowvec gamma_2(q);
  arma::rowvec gamma_new(q);
  gamma_1 = Gamma.row(id1-1);
  gamma_2 = Gamma.row(id2-1);
  // Find those groups that are associated with the kth mixture
  int n1=getind(R,id1).n_cols;
  int n2=getind(R,id2).n_cols;
  arma::mat Z1(n,q);
  arma::mat Y1(n,n);
  arma::mat Z2(n,q);
  arma::mat Y2(n,n);
  arma::rowvec ind1(m);
  arma::rowvec ind2(m);
  int nid = n1+n2;
  if(n1>0){
    ind1.set_size(n1);
    ind1 = getind(R,id1);
    Z1.set_size(n1,q);
    Y1.set_size(n,n1);
    Z1 = rowsome( Z, ind1);
    Y1 = colsome(Y, ind1);
  }
  if(n2>0){
    ind2.set_size(n2);
    ind2 = getind(R,id2);
    Z2.set_size(n2,q);
    Y2.set_size(n,n2);
    Z2 = rowsome(Z,ind2);
    Y2 = colsome(Y, ind2);
  }
  
  //re-arrange labels
  arma::rowvec R_new(m);
  R_new = R;
  for(int i=0; i<m; i++){
    if((R(i)>id1) & (R(i)<id2)){
      R_new(i)= R(i) -1;
    }
    if((R(i)>id2)){
      R_new(i)= R(i) -2;
    }
    if((R(i)==id1) || (R(i)==id2)){
      R_new(i) = Knew;
    }
  }
  
  //Compute the merge transformation using moment-matching principle
  double w1_new = w1 + w2;
  arma::rowvec w_new(Knew);
  w_new(Knew - 1) = w1_new;
  for(int i=0; i<Knew-1; i++){
    if(i<id1-1){
      w_new(i) = w(i);
    }
    if((i>=id1-1) & (i<id2-1)){
      w_new(i) = w(i+1);
    }
    if(i>=id2-1){
      w_new(i) = w(i+2);
    }
  }
  
  double alpha = w1/w1_new;
  arma::mat Gamma_new(Knew,q);
  gamma_new = (w1 * gamma_1 + w2 * gamma_2)/w1_new;
  Gamma_new = rbind0(removeiii(removeiii(Gamma,id1), id2-1),gamma_new);
  
  arma::rowvec beta(q);
  double ratio_propose = 0;
  for(int i=0; i<q; i++){
    beta(i) = (gamma_new(i) - gamma_1(i)) * std::sqrt(alpha/(1-alpha));
    if((beta(i)<1) & (beta(i)>0)){
      ratio_propose += R::dbeta(beta(i),2,2,true);
    }
  }
  //compute the proposal ratio
  double diff = 0;
  arma::mat temp(1,1);
  if(n1>0){
    for(int i=0; i<n1; i++){
      pro = pro_init;
      temp(0,0) = sum(gamma_1 % Z1.row(i));
      pro(0) += dmvnrm_arma(Y1.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      temp(0,0) = sum(gamma_2 % Z1.row(i));
      pro = pro -max(pro);
      pro = exp(pro);
      pro = pro/sum(pro);
      diff = pro(1) - pro(0);
      pro(0) += 0.00001*diff;
      pro(1) -= 0.00001*diff;
      ratio_propose += log(pro(0));
    }
  }
  if(n2>0){
    for(int i=0; i<n2; i++){
      pro = pro_init;
      temp(0,0) = sum(gamma_1 % Z2.row(i));
      pro(0) += dmvnrm_arma(Y2.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      temp(0,0) = sum(gamma_2 % Z2.row(i));
      pro(1) += dmvnrm_arma(Y2.col(i).t() ,repmat(temp,1,n), sigma_square,true);
      pro = pro -max(pro);
      pro = exp(pro);
      pro = pro/sum(pro);
      diff = pro(1) - pro(0);
      pro(0) += 0.00001*diff;
      pro(1) -= 0.00001*diff;
      ratio_propose += log(pro(1));
    }
  }
  
  //compute the log Jacobian of moment-matching transformation
  double log_Jacobian = log(w1_new) - (q/2.0) * log(alpha*(1-alpha));
  //compute the likelihood ratio
  double likeli_now = 0;
  double likeli_new = 0;
  
  if(n1>0){
    arma::rowvec mu_new1(n1);
    arma::rowvec mu_now1(n1);
    arma::rowvec indT1(n1);
    mu_now1 = colsome(mu,ind1);
    mu_new1 = gamma_new * Z1.t();
    for(int i=0; i<n1; i++){
      indT1(i) = ceil((ind1(i) + 1)/6.0) - 1;
    }
    for(int i=0; i<n; i++){
      likeli_now += dmvnrm_arma(Y1.row(i) ,mu_now1,  sigma_square,true);
      likeli_new += dmvnrm_arma(Y1.row(i) ,mu_new1,  sigma_square,true);
    }
    
    
  }
  if(n2>0){
    arma::rowvec mu_new2(n2);
    arma::rowvec mu_now2(n2);
    arma::rowvec indT2(n2);
    mu_now2 = colsome(mu,ind2);
    mu_new2 = gamma_new * Z2.t();
    for(int i=0; i<n2; i++){
      indT2(i) = ceil((ind2(i) + 1)/6.0) - 1;
    }
    for(int i=0; i<n; i++){
      likeli_now += dmvnrm_arma(Y2.row(i) ,mu_now2, sigma_square,true);
      likeli_new += dmvnrm_arma(Y2.row(i) ,mu_new2, sigma_square,true);
    }
  }
  
  double ratio_like = likeli_new - likeli_now;
  
  //compute the prior ratio
  double ratio_prior_w = (hyper_delta - 1 + n1) * log(w1) + (hyper_delta - 1 + n2) * log(w2) - 
    (hyper_delta - 1 + nid) * log(w1_new) - log(R::beta(hyper_delta, Knew * hyper_delta));
  arma::mat C_new(Knew, Knew);
  C_new = updateC(Gamma_new,theta,tau);
  double ratio_prior_gamma = log(det(C)) - log(det(C_new));
  double ratio_prior = ratio_prior_w + ratio_prior_gamma;
  double death_ratio = log(K)*2 + ratio_propose + ratio_like - ratio_prior - log_Jacobian;
  if(Knew ==1){
    death_ratio = death_ratio + log(2);
  }
  
  if(log(R::runif(0,1)) < death_ratio){
    K=Knew;
    Gamma = Gamma_new;
    R = R_new;
    C = C_new;
    w = w_new;
  }
  
  ans["K"] = K;
  ans["w"] = w;
  ans["Gamma"] = Gamma;
  ans["R"]= R;
  ans["C"] = C;
  return (ans);
}

// [[Rcpp::export]]
List RJi(arma::rowvec w,
         int K,
         arma::mat Gamma,
         arma::rowvec Beta,
         arma::mat X,
         arma::mat Y,
         arma::mat Z,
         arma::rowvec R,
         arma::mat delta,
         arma::rowvec mu,
         arma::mat mu_star,
         double c,
         double sigma_square,
         arma::mat C,
         double theta, double tau,
         int m, int n, int q, int T0, double hyper_delta = 1){
  arma::mat diff(n,m);
  diff = arma::repmat(X*Beta.t(),1,m);
  Y = Y - diff;
  mu = mu - diff.row(0);
  double merge_or_split = R::runif(0,1);
  if (((K == 1) || (merge_or_split < 0.5)) & (K<10)){
    return (
        Split(w, K, Gamma, Beta, X, Y, Z,
              R, delta, mu, mu_star, c, sigma_square, 
              C, theta, tau, m, n, q, T0, hyper_delta)
    );
  }
  else{
    return (
        Merge(w, K, Gamma, Beta, X, Y, Z,
              R, delta, mu, mu_star, c, sigma_square, 
              C, theta, tau, m, n, q, T0, hyper_delta)
    );
  }
}



// [[Rcpp::export]]
List RJinomiss(arma::rowvec w,
               int K,
               arma::mat Gamma,
               arma::rowvec Beta,
               arma::mat X,
               arma::mat Y,
               arma::mat Z,
               arma::rowvec R,
               arma::mat delta,
               arma::rowvec mu,
               double sigma_square,
               arma::mat C,
               double theta, double tau,
               int m, int n, int q, int T0, double hyper_delta = 1){
  arma::mat diff(n,m);
  diff = arma::repmat(X*Beta.t(),1,m);
  Y = Y - diff;
  mu = mu - diff.row(0);
  double merge_or_split = R::runif(0,1);
  if (((K == 1) || (merge_or_split < 0.5)) & (K<10)){
    return (
        Splitnomiss(w, K, Gamma, Beta, X, Y, Z,
                    R, delta, mu,  sigma_square, 
                    C, theta, tau, m, n, q, T0, hyper_delta)
    );
  }
  else{
    return (
        Mergenomiss(w, K, Gamma, Beta, X, Y, Z,
                    R, delta, mu, sigma_square, 
                    C, theta, tau, m, n, q, T0, hyper_delta)
    );
  }
}


// [[Rcpp::export]]
arma::rowvec updateRi(arma::rowvec w,
                      arma::mat Gamma,
                      arma::rowvec Beta,
                      arma::mat Y,
                      arma::mat Z,
                      arma::mat delta,
                      arma::rowvec mu,
                      arma::mat mu_star,
                      double c,
                      double sigma_square,
                      int K, arma::mat X,
                      int m, int n, int q){
  arma::rowvec R(m);
  arma::mat diff(n,m);
  diff = arma::repmat(X*Beta.t(),1,m);
  Y = Y - diff;
  mu = mu - diff.row(0);
  double mu_s, va, likeli;
  int T_now;
  arma::mat mu_new(1,1);
  arma::rowvec pro(K);
  arma::colvec mu_star_s(n);
  arma::colvec mu_star_new_s(n);
  arma::rowvec Gammai(q);
  for(int s=0; s<m; s++){
    mu_s = mu(s);
    T_now = ceil((s + 1)/6.0) - 1;
    mu_star_s = mu_star.col(T_now);
    pro = log(w)/log_e;
    for(int k=0; k<K; k++){
      Gammai = Gamma.row(k);
      mu_new(0,0) = sum(Gammai % Z.row(s));
      mu_star_new_s = mu_star_s + c*(mu_new(0,0) - mu_s)*(1.0/6.0);
      likeli = 0;
      likeli += dmvnrm_arma(Y.col(s).t(), arma::repmat(mu_new,1,n), sigma_square, true);
      for(int i=0; i<n; i++){
        va = min(5.0, mu_star_new_s(i));
        va = max(va, -5.0);
        likeli += R::pnorm(va,0,1,true, true)* delta(i,T_now) + log(1-R::pnorm(va,0,1,true, false))*(1-delta(i,T_now));
      }
      pro(k) += likeli;
    }
    pro = pro -max(pro);
    pro = exp(pro);
    pro = pro/sum(pro);
    //assign indicator
    
    R(s) =  rmunoim(pro);
  }
  return (R);
}

// [[Rcpp::export]]
arma::rowvec updateRinomiss(arma::rowvec w,
                            arma::mat Gamma,
                            arma::rowvec Beta,
                            arma::mat Y,
                            arma::mat Z,
                            arma::mat delta,
                            arma::rowvec mu,
                            double sigma_square,
                            int K, arma::mat X,
                            int m, int n, int q){
  arma::rowvec R(m);
  arma::mat diff(n,m);
  diff = arma::repmat(X*Beta.t(),1,m);
  Y = Y - diff;
  mu = mu - diff.row(0);
  double mu_s,  likeli;
  arma::mat mu_new(1,1);
  arma::rowvec pro(K);
  arma::rowvec Gammai(q);
  for(int s=0; s<m; s++){
    mu_s = mu(s);
    pro = log(w)/log_e;
    for(int k=0; k<K; k++){
      Gammai = Gamma.row(k);
      mu_new(0,0) = sum(Gammai % Z.row(s));
      likeli = 0;
      likeli += dmvnrm_arma(Y.col(s).t(), arma::repmat(mu_new,1,n), sigma_square, true);
      pro(k) += likeli;
    }
    pro = pro -max(pro);
    pro = exp(pro);
    pro = pro/sum(pro);
    //assign indicator
    
    R(s) =  rmunoim(pro);
  }
  return (R);
}

// [[Rcpp::export]]
arma::rowvec sample_int(int size, arma::rowvec pro){
  arma::rowvec ans(size);
  for(int i=0; i<size; i++){
    ans(i) = rmunoim(pro);
  }
  return (ans);
}


// [[Rcpp::export]]
arma::mat updateR(arma::mat w,
                  arma::cube Gamma,
                  arma::mat Beta,
                  arma::mat Y,
                  arma::mat Z,
                  arma::mat delta,
                  arma::mat mu,
                  arma::mat mu_star,
                  double c, int S,
                  double sigma_square,
                  arma::rowvec K, 
                  arma::rowvec E,
                  arma::mat X,
                  int m, int n, int q, int p, int T0){
  arma::mat R(S,m);
  arma::rowvec wi(10);
  arma::mat Gammai(10,q);
  arma::rowvec Betai(p);
  arma::rowvec indi(n);
  arma::mat Yi(n,m);
  arma::mat deltai(n,T0);
  arma::rowvec mui(m);
  arma::mat mu_stari(n,T0);
  arma::rowvec Ri(m);
  arma::mat Xi(n,p);
  
  int len, Ki;
  for(int i=0; i<S; i++){
    wi.set_size(1,K(i));
    wi = colsome(w.row(i),ton(K(i)));
    len=getind(E,i+1).n_cols;
    if(len==0){
      R.row(i) = sample_int(m, wi);
    }
    else{
      Gammai.set_size(K(i),q);
      Gammai = getGammai(Gamma,i,K(i),q);
      Betai = Beta.row(i);
      indi.set_size(1,len);
      indi = getind(E,i+1);
      Yi.set_size(len,m);
      Yi = rowsome(Y,indi);
      deltai.set_size(len,T0);
      deltai = rowsome(delta,indi);
      mui = mu.row(indi(0));
      mu_stari.set_size(len,T0);
      mu_stari = rowsome(mu_star, indi);
      Ki = K(i);
      Xi.set_size(len,p);
      Xi = rowsome(X,indi);
      R.row(i) = updateRi(wi,Gammai, Betai, Yi, Z, deltai, mui, mu_stari, c, sigma_square, Ki,
            Xi, m, len, q);
    }
    
    
  }
  return (R);
}

// [[Rcpp::export]]
arma::mat updateRnomiss(arma::mat w,
                        arma::cube Gamma,
                        arma::mat Beta,
                        arma::mat Y,
                        arma::mat Z,
                        arma::mat delta,
                        arma::mat mu,
                        int S,
                        double sigma_square,
                        arma::rowvec K, 
                        arma::rowvec E,
                        arma::mat X,
                        int m, int n, int q, int p, int T0){
  arma::mat R(S,m);
  arma::rowvec wi(10);
  arma::mat Gammai(10,q);
  arma::rowvec Betai(p);
  arma::rowvec indi(n);
  arma::mat Yi(n,m);
  arma::mat deltai(n,T0);
  arma::rowvec mui(m);
  arma::rowvec Ri(m);
  arma::mat Xi(n,p);
  
  int len, Ki;
  for(int i=0; i<S; i++){
    wi.set_size(1,K(i));
    wi = colsome(w.row(i),ton(K(i)));
    len=getind(E,i+1).n_cols;
    if(len==0){
      R.row(i) = sample_int(m, wi);
    }
    else{
      Gammai.set_size(K(i),q);
      Gammai = getGammai(Gamma,i,K(i),q);
      Betai = Beta.row(i);
      indi.set_size(1,len);
      indi = getind(E,i+1);
      Yi.set_size(len,m);
      Yi = rowsome(Y,indi);
      deltai.set_size(len,T0);
      deltai = rowsome(delta,indi);
      mui = mu.row(indi(0));
      Ki = K(i);
      Xi.set_size(len,p);
      Xi = rowsome(X,indi);
      R.row(i) = updateRinomiss(wi,Gammai, Betai, Yi, Z, deltai, mui,  sigma_square, Ki,
            Xi, m, len, q);
    }
    
    
  }
  return (R);
}
// [[Rcpp::export]]
arma::mat updatemustar(arma::mat mu,
                       arma::rowvec c,
                       int n, int T0, arma::mat D){
  arma::mat mustar(n,T0);
  arma::vec temp2(T0);
  for(int i=0; i<n;i++){
    temp2 = cDmu(c,D*mu.row(i).t());
    mustar.row(i) = temp2.t();
  }
  return (mustar);
}



// [[Rcpp::export]]
double rtrun_norm( double mu, double sigma, double a, double b){
  double alpha;
  double alpha_cdf;
  double beta;
  double beta_cdf;
  double u;
  double x;
  double xi;
  double xi_cdf;
  
  alpha = ( a - mu ) / sigma;
  beta = ( b - mu ) / sigma;
  alpha_cdf = R::pnorm(alpha,0,1,true,false);
  beta_cdf = R::pnorm(beta,0,1,true,false);
  if(alpha_cdf>0.9999){
    return (R::runif(a,a+sigma));
  }else if(beta_cdf<0.0001){
    return (R::runif(b-sigma,b));
  }else{
    u = R::runif(0,1);
    xi_cdf = alpha_cdf + u * ( beta_cdf - alpha_cdf );
    xi = R::qnorm5(xi_cdf,0,1,true,false);
    x = mu + sigma * xi;
    
    return (x);
  }
  
}

// [[Rcpp::export]]
arma::mat updateZstar(arma::mat mu_star, arma::mat delta,
                      int n, int T0){
  arma::mat Z(n,T0);
  for(int i=0; i<n; i++){
    for(int t=0; t<T0; t++){
      if(delta(i,t) == 0){
        Z(i,t) = rtrun_norm(mu_star(i,t),1,-Inf,0);
      }else{
        Z(i,t) = rtrun_norm(mu_star(i,t),1,0,Inf);
      }
    }
  }
  return (Z);
}

// [[Rcpp::export]]
double updatec(arma::mat Zstar, arma::mat mu, arma::mat D,
               double sigmac, int n, int T0){
  double sumxx = 0;
  double sumxy = 0;
  arma::rowvec xi(T0);
  arma::rowvec Yc(T0);
  for(int i=0; i<n; i++){
    xi = (D * mu.row(i).t()).t();
    Yc = Zstar.row(i);
    
    sumxx += sum(xi % xi );
    sumxy += sum(xi % Yc);
  }
  double sigman = 1.0/(1/sigmac + sumxx);
  double betan = sigman * sumxy;
  double ans = R::rnorm(betan,sqrt(sigman));
  return (ans);
  
}

// [[Rcpp::export]]
List updateBetanoDPP(arma::mat X,
                     arma::mat Y,
                     arma::mat Z,
                     arma::mat delta,
                     arma::mat Beta,
                     arma::cube Gamma,
                     arma::rowvec E,
                     arma::mat R,
                     double S,
                     arma::rowvec Ds,
                     arma::mat mustar,
                     arma::mat mu,
                     double sigma,
                     arma::rowvec c,
                     arma::mat step,
                     arma::mat runif,
                     int n, int m, int T0, int p, int q, arma::mat D){
  List ans;
  arma::rowvec Betanew(p);
  arma::vec mustarnewj(T0);
  arma::vec munewj(m);
  arma::mat mustarnew(n,T0);
  arma::mat munew(n,m);
  arma::mat Ybeta(n,m);
  arma::rowvec indj(m);
  arma::mat Zj(m,q);
  arma::rowvec gammaij(q);
  arma::rowvec betai(p);
  arma::rowvec indi(n);
  arma::vec beta0(m);
  int l;
  double likelinow, likelinew,va;
  munew = mu;
  mustarnew = mustar;
  Ybeta = Y;
  for(int i=0; i<n; i++){
    for(int j=1; j<=Ds(E(i)-1);j++){
      l=getind(R.row(E(i)-1),j).n_cols;
      indj.reshape(1,l);
      indj = getind(R.row(E(i)-1),j);
      Zj.reshape(l,q);
      Zj = rowsome(Z,indj);
      gammaij = Gamma.slice(E(i)-1).row(j-1);
      Ybeta = minusvalue(Ybeta,i,indj, (Zj*gammaij.t()).t());
    }
  }
  //now update beta one by one row
  
  for(int i=0; i<S;i++){
    likelinow = 0;
    betai = Beta.row(i);
    l=getind(E,i+1).n_cols;
    indi.reshape(1,l);
    indi = getind(E,i+1);
    for(int j=0; j<l; j++){
      //j in R is indi(j) now
      beta0 = matrix(X.row(indi(j)),m) * betai.t();
      likelinow += dmvnrm_arma(Ybeta.row(indi(j)),  beta0.t(),   sigma , true);
      for(int t=0; t<T0; t++){
        va = min(5.0, mustar(indi(j),t));
        va = max(va, -5.0);
        likelinow += R::pnorm(va,0,1,true, true)* delta(indi(j),t) + log(1-R::pnorm(va,0,1,true, false))*(1-delta(indi(j),t));
      }
    }
    //update Beta one by one
    
    for(int j=0; j<p;j++){
      Betanew = Beta.row(i);
      Betanew(j) = Betanew(j) + step(i,j);
      likelinew = 0;
      for(int k=0; k<l;k++){
        //k in R is now indi(k)
        beta0 = matrix(X.row(indi(k)),m)*Betanew.t();
        likelinew += dmvnrm_arma(Ybeta.row(indi(k)),  beta0.t(), sigma , true);
        munewj = mu.row(indi(k)).t() + matrix(X.row(indi(k)),m) * (Betanew.t()-Beta.row(i).t());
        mustarnewj = mustar.row(indi(k)).t() + cDmu(c,D*(munewj)) - cDmu(c,D*mu.row(indi(k)).t());
        mustarnew.row(indi(k)) = mustarnewj.t();
        munew.row(indi(k)) = munewj.t();
        for(int t=0; t<T0; t++){
          va = min(5.0, mustarnewj(t));
          va = max(va, -5.0);
          likelinew += R::pnorm(va,0,1,true,true)*delta(indi(k),t) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(indi(k),t));
        }
      }
      //compute prior
      if(l==0){
        likelinew = 0;
        likelinow = 0;
      }
      if((likelinew - likelinow + R::dnorm4(Betanew(j),0,100,true) - R::dnorm4(Beta(i,j),0,100,true))>log(runif(i,j))){
        likelinow = likelinew;
        Beta.row(i) = Betanew;
        mustar = mustarnew;
        mu = munew;
      }
    }
  }
  ans["Beta"] = Beta;
  ans["mu"]= mu;
  ans["mustar"] = mustar;
  return (ans);
}


// [[Rcpp::export]]
List updateGammanoDPP(arma::mat X,
                      arma::mat Y,
                      arma::mat Z,
                      arma::mat delta,
                      arma::mat Beta,
                      arma::cube Gamma,
                      arma::rowvec E,
                      arma::mat R,
                      double S,
                      arma::rowvec Ds,
                      arma::mat mu,
                      arma::mat mustar,
                      double sigma,
                      arma::rowvec c,
                      arma::cube step,
                      arma::cube runif,
                      int n, int m, int T0, int p, int q, arma::mat D){
  List ans;
  arma::mat mustarnew(n, T0);
  arma::mat munew(n, m);
  arma::rowvec inds(n);
  arma::rowvec indi(m);
  arma::rowvec indT(T0);
  arma::mat Gammai(max(Ds),q);
  mustarnew = mustar;
  arma::rowvec temp(T0);
  munew =mu;
  arma::mat temp1(1,1);
  arma::rowvec b(9);
  arma::rowvec temp2(10);
  arma::rowvec bnew(9);
  arma::vec beta0(10);
  arma::rowvec Ygamma(m);
  arma::mat Zi(m,q);
  arma::rowvec gammanew(q);
  //arma::cube L(max(Ds),q,S);
  //arma::cube P(max(Ds),q,S);
  int lens, ns, nt;
  double likelinow, likelinew, xbeta, va;
  for(int i=0; i<S; i++){
    //cout<<i<<endl;
    lens=getind(E,i+1).n_cols;
    inds.reshape(1,lens);
    inds = getind(E,i+1);
    Gammai.reshape(Ds(i),q);
    Gammai = getGammai(Gamma,i,Ds(i),q);
    for(int j=0; j<Ds(i); j++){
      //cout<<j<<endl;
      ns = getind(R.row(i),j+1).n_cols;
      indi.reshape(1,ns);
      indi = getind(R.row(i),j+1);
      beta0.reshape(ns,1);
      Zi.reshape(ns,q);
      temp.reshape(1,ns);
      for(int k = 0; k<ns; k++){
        //k in R is indi(k) now
        temp(k) =  findT(D.col(indi(k)).t());
      }
      nt = Nuni(temp);
      indT.reshape(1,nt);
      indT = unique(temp);
      likelinow = 0;
      for(int k = 0; k<lens;k++){
        //k in R is now inds(k)
        xbeta = sum(X.row(inds(k))%Beta.row(i));
        Ygamma = Y.row(inds(k)) -  xbeta;
        beta0 = rowsome(Z,indi)* Gammai.row(j).t();
        likelinow += dmvnrm_arma(colsome(Ygamma,indi), beta0.t(), sigma, true);
        for(int t=0; t<nt; t++){
          //t in R is now indT(t)
          va = min(5.0, mustar(inds(k),indT(t)));
          va = max(va, -5.0);
          likelinow += R::pnorm(va,0,1,true, true)*delta(inds(k),indT(t)) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(inds(k),indT(t)));
        }
      }
      //update Gamma one element by one element
      for(int l=0; l<q; l++){
        gammanew = Gammai.row(j);
        gammanew(l) = gammanew(l) + step(j,l,i);
        likelinew = 0;
        for(int k=0; k<lens; k++){
          //k in R is now inds(k)
          xbeta = sum(X.row(inds(k))%Beta.row(i));
          Ygamma = Y.row(inds(k)) -  xbeta;
          beta0 = rowsome(Z,indi)* gammanew.t();
          likelinew += dmvnrm_arma(colsome(Ygamma,indi), beta0.t(), sigma, true);
          munew = putvalue(munew,inds(k),indi, 
                           computemu(mu,inds(k),indi, rowsome(Z,indi), Gammai.row(j), gammanew));
          mustarnew = putvalue(mustarnew,inds(k),indT, 
                               computemustar(mustar, inds(k),indT, c, D, mu.row(inds(k)), munew.row(inds(k)), T0));
          for(int t=0; t<nt; t++){
            //t in R is now indT(t)
            va = min(5.0, mustarnew(inds(k),indT(t)));
            va = max(va, -5.0);
            likelinew += R::pnorm(va,0,1,true, true)*delta(inds(k),indT(t)) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(inds(k),indT(t)));
            
          }
          
        }
        if(lens == 0){
          likelinew =0;
          likelinow = 0;
        }
        if(ns == 0){
          likelinew = 0;
          likelinow = 0;
        }
        if((R::dnorm4(gammanew(l),0,100,true)- R::dnorm4(Gammai(j,l),0,100,true) + likelinew - likelinow)>log(runif(j,l,i))){
          mu = munew;
          mustar = mustarnew;
          likelinow = likelinew;
          Gammai.row(j) = gammanew;
        }
        else{
          munew = mu;
          mustarnew = mustar;
        }
      }
    }
    //Gamma[,,i] = Gammai
    Gamma = putGamma(Gamma, Gammai, i, q);
  }
  ans["mu"]= mu;
  ans["mustar"] = mustar;
  ans["Gamma"]= Gamma;
  //ans["L"] = L;
  //ans["P"] = P;
  return(ans);
}


arma::cube spcopyGamma2(arma::cube Gamma, arma::rowvec Ds, int S, int q){
  int max0;
  max0 = max(Ds);
  arma::cube ans(max0,q,S);
  ans.fill(0);
  for(int i=0; i<S; i++){
    for(int j=0; j<Ds(i); j++){
      ans.slice(i).row(j) = Gamma.slice(i).row(j);
    }
  }
  return (ans);
}

arma::cube spcopyGamma(arma::cube Gamma, int k, arma::mat Gammai, arma::rowvec Ds, int S, int q){
  //Gamma[,,i]<-Gammai
  //need to change dim
  //k starts from 0
  int max0;
  max0 = max(Ds);
  arma::cube ans(max0,q,S);
  ans.fill(0);
  for(int i=0; i<S; i++){
    if(i==k){
      for(int j=0; j<Ds(i); j++){
        ans.slice(i).row(j) = Gammai.row(j);
      }
    }
    else{
      for(int j=0; j<Ds(i); j++){
        ans.slice(i).row(j) = Gamma.slice(i).row(j);
      }
    }
  }
  return (ans);
}




// [[Rcpp::export]]
List updateRnoDPP(arma::mat Beta,
                  arma::cube Gamma,
                  arma::mat X,
                  arma::mat Y,
                  arma::mat Z,
                  arma::mat delta,
                  arma::rowvec E,
                  arma::mat R,
                  int S,
                  arma::rowvec Ds,
                  arma::mat mu,
                  arma::mat mustar,
                  double sigma,
                  arma::rowvec Ms,
                  int n, int m, int T0, int p, int q, arma::mat D, int Dmax=10){
  List ans;
  arma::rowvec Ns(S);
  Ns.fill(0);
  for(int i=0;i<n;i++){
    Ns(E(i)-1)++;
  }
  arma::rowvec indi(n);
  arma::rowvec ns(m);
  arma::mat Yr = Y;
  double likeli, mu0now, mu0, mustarnew, va;
  int T1, max1, nnow;
  arma::rowvec pro(10);
  arma::vec temp1(m);
  arma::mat Gammai(10,q);
  arma::rowvec temp2(10);
  arma::mat temp3(max(Ds),q);
  arma::rowvec gammanew(q);
  arma::cube temp4=Gamma;
  for(int i=0; i<n; i++){
    temp1 = matrix(X.row(i),m)*Beta.row(E(i)-1).t();
    Yr.row(i) = Yr.row(i) - temp1.t();
  }
  for(int i=0; i<S; i++){
    Gammai.set_size(Ds(i),q);
    Gammai = getGammai(Gamma,i,Ds(i),q);
    indi.set_size(1,Ns(i));
    indi = getind(E,i+1);
    ns.set_size(1,Ds(i));
    pro.set_size(1,Ds(i)+1);
    for(int j=0; j<Ds(i); j++){
      ns(j) = sum(R.row(i)==(j+1));
    }
    for(int j=0; j<m; j++){
      //find the corresponding teeth
      T1 = findT(D.col(j).t());
      mu0now = sum(Z.row(j)%Gammai.row(R(i,j)-1));
      ns(R(i,j)-1)--;
      if(ns(R(i,j)-1)==0){
        Ds(i)--;
        if(R::runif(0,1)<((Ds(i)-1)/Ds(i))){
          ns(R(i,j)-1)++;
          Ds(i)++;
          continue;
        }
        temp2.set_size(1,Ds(i));
        temp2 = removei(ns, R(i,j));
        ns.set_size(1,Ds(i));
        pro.set_size(1,Ds(i)+1);
        ns = temp2;
        temp3.set_size(Ds(i),q);
        temp3 = removeiii(Gammai,R(i,j));
        Gammai.set_size(Ds(i),q);
        Gammai = temp3;
        for(int k=0; k<m; k++){
          if(R(i,j)<R(i,k)){
            R(i,k)--;
          }
        }
      }
      //compute prob
      pro= connect0(ns,Ms(i)/(Ds(i)+1));
      pro = log(pro);
      for(int k=0; k<Ds(i); k++){
        likeli = 0;
        mu0 = sum(Z.row(j)%Gammai.row(k));
        for(int l=0; l<Ns(i); l++){
          //l in R is now indi(l)
          if(Yr(indi(l),j)==Yr(indi(l),j)){
            likeli+= R::dnorm(Yr(indi(l),j),mu0, sigma, true);
          }
          
          mustarnew = mustar(indi(l), T1) - mu0now/6 + mu0/6;
          va = min(5.0, mustarnew);
          va = max(va, -5.0);
          likeli += R::pnorm(va,0,1,true,true)*delta(indi(l),T1) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(indi(l),T1));
        }
        pro(k) += likeli;
      }
      //for new cluster
      likeli = 0;
      for(int k=0; k<q; k++){
        gammanew(k) = R::rnorm(R::runif(-5,5), sqrt(5));
      }
      mu0=sum(Z.row(j)%gammanew);
      for(int l=0; l<Ns(i);l++){
        if(Yr(indi(l),j)==Yr(indi(l),j)){
          likeli+= R::dnorm(Yr(indi(l),j),mu0, sigma, true);
        }
        mustarnew = mustar(indi(l), T1) - mu0now/6 + mu0/6;
        va = min(5.0, mustarnew);
        va = max(va, -5.0);
        likeli += R::pnorm(va,0,1,true,true)*delta(indi(l),T1) + log(1-R::pnorm(va,0,1,true,false))*(1-delta(indi(l),T1));
      }
      pro(Ds(i)) += likeli;
      //compute the prior
      if(Ds(i)==Dmax){
        pro.reshape(1,Dmax);
      }
      pro = pro -max(pro);
      pro = exp(pro);
      pro = pro/sum(pro);
      //assign indicator
      R(i,j) =  rmunoim(pro);
      if(R(i,j)==(Ds(i)+1)){
        Ds(i)++;
        temp2.set_size(1,Ds(i));
        temp2 = connect0(ns,1);
        ns.set_size(1,Ds(i));
        ns = temp2;
        temp3.set_size(Ds(i),q);
        temp3 = rbind0(Gammai, gammanew);
        Gammai.set_size(Ds(i),q);
        Gammai = temp3;
      }
      else{
        ns(R(i,j)-1)++;
      }
    }
    //put Gammai back to Gamma
    nnow = Gamma.slice(0).n_rows;
    if(Ds(i)>nnow){
      temp4.set_size(Ds(i),q,S);
      temp4 = spcopyGamma(Gamma,i,Gammai,Ds,S,q);
      Gamma.set_size(Ds(i),q,S);
      Gamma = temp4;
    }
    else{
      Gamma = putGamma(Gamma, Gammai, i, q);
    }
  }
  max1 = max(Ds);
  nnow = Gamma.slice(0).n_rows;
  if(max1<nnow){
    temp4.set_size(max1,q,S);
    temp4 = spcopyGamma2(Gamma,Ds,S,q);
    Gamma.set_size(max1,q,S);
    Gamma = temp4;
  }
  ans["Gamma"] = Gamma;
  ans["Ds"] = Ds;
  ans["R"] = R;
  ans["Yr"] = Yr;
  return (ans);
}

// [[Rcpp::export]]
List Split_empty(arma::rowvec w,
                 int K,
                 arma::mat Gamma,
                 arma::rowvec R,
                 arma::mat C,
                 double theta, double tau,
                 int m, int q, double hyper_delta = 1){
  List ans;
  int Knew = K + 1;
  int l = sampleint(K);
  double w_1 = w(l-1);
  int nid;
  arma::rowvec gamma_1(q);
  nid=getind(R,l).n_cols;
  arma::rowvec l_label(nid);
  l_label = getind(R,l);
  
  //re-arrange labels
  arma::rowvec R_new(m);
  
  for(int i=0; i<m; i++){
    if(R(i)>l){
      R_new(i) = R(i)-1;
    }
    else{
      R_new(i) = R(i);
    }
  }
  //compute the split transformation using moment matching principle
  arma::rowvec beta(q);
  for(int i=0; i<q; i++){
    beta(i) = R::rbeta(2,2);
  }
  double alpha = R::runif(0,1);
  double w_new_1 = w_1 * alpha;
  double w_new_2 = w_1 * (1-alpha);
  arma::rowvec pro(2);
  pro(0) = w_new_1;
  pro(1) = w_new_2;
  arma::rowvec gamma_new_1(q);
  arma::rowvec gamma_new_2(q);
  double ratio_propose = 0;
  for(int i=0; i<q; i++){
    gamma_new_1(i) = gamma_1(i) - std::sqrt((1-alpha)/alpha) * beta(i);
    gamma_new_2(i) = gamma_1(i) + std::sqrt(alpha/(1-alpha)) * beta(i);
    ratio_propose += R::dbeta(beta(i),2,2,true);
  }
  
  //compute the proposal ratio
  int n1 = 0;
  int n2 = 0;
  if(nid>0){
    for(int i=0; i<nid; i++){
      if(R::runif(0,1)<pro(0)){
        ratio_propose += log(pro(0));
        n1 += 1;
        R_new(l_label(i)) = K;
      }
      else{
        ratio_propose += log(pro(1));
        n2 += 1;
        R_new(l_label(i)) = Knew;
      }
    }
    
  }
  
  //compute the log Jacobian
  double log_Jacobian = log(w_1) - (q/2.0) * log(alpha*(1-alpha));
  
  //compute the prior ratio
  double ratio_prior_w = (hyper_delta - 1 + n1) * log(w_new_1) + (hyper_delta - 1 + n2) * log(w_new_2) - 
    (hyper_delta - 1 + nid) * log(w_1) - log(R::beta(hyper_delta, K * hyper_delta));
  
  arma::mat Gamma_new(Knew,q);
  Gamma_new = rbind(removeiii(Gamma, l),gamma_new_1, gamma_new_2);
  arma::mat C_new(Knew,Knew);
  C_new = updateC(Gamma_new,theta,tau);
  double ratio_prior_gamma = log(det(C_new)) - log(det(C));
  double ratio_prior = ratio_prior_gamma + ratio_prior_w;
  double birth_ratio = ratio_prior - log(Knew) + log_Jacobian - log(Knew) - ratio_propose;
  if(Knew == 2){
    birth_ratio -= log(2); 
  }
  if(log(R::runif(0,1)) < birth_ratio){
    K=Knew;
    Gamma = Gamma_new;
    R = R_new;
    C = C_new;
    w = connect(removei(w,l), w_new_1, w_new_2);
  }
  
  ans["K"] = K;
  ans["w"] = w;
  ans["Gamma"] = Gamma;
  ans["R"]= R;
  ans["C"] = C;
  return (ans);
}

// [[Rcpp::export]]
List Merge_empty(arma::rowvec w,
                 int K,
                 arma::mat Gamma,
                 arma::rowvec R,
                 arma::mat C,
                 double theta, double tau,
                 int m,  int q, double hyper_delta = 1){
  List ans;
  int Knew = K-1;
  
  //Uniformly pick two labels that are potentially to be merged
  arma::rowvec id(2);
  id = sampleint2(K);
  int id1 = id(0);
  int id2 = id(1);
  double w1 = w(id1-1);
  double w2 = w(id2-1);
  arma::rowvec pro_init(2);
  arma::rowvec pro(2);
  pro_init(0) = w1;
  pro_init(1) = w2;
  arma::rowvec gamma_1(q);
  arma::rowvec gamma_2(q);
  arma::rowvec gamma_new(q);
  gamma_1 = Gamma.row(id1-1);
  gamma_2 = Gamma.row(id2-1);
  
  // Find those groups that are associated with the kth mixture
  int n1=getind(R,id1).n_cols;
  int n2=getind(R,id2).n_cols;
  int nid = n1 + n2;
  arma::rowvec ind1(m);
  arma::rowvec ind2(m);
  if(n1>0){
    ind1.set_size(n1);
    ind1 = getind(R,id1);
  }
  if(n2>0){
    ind2.set_size(n2);
    ind2 = getind(R,id2);
  }
  
  //re-arrange labels
  arma::rowvec R_new(m);
  R_new = R;
  for(int i=0; i<m; i++){
    if((R(i)>id1) & (R(i)<id2)){
      R_new(i)= R(i) -1;
    }
    if((R(i)>id2)){
      R_new(i)= R(i) -2;
    }
    if((R(i)==id1) || (R(i)==id2)){
      R_new(i) = Knew;
    }
  }
  
  //Compute the merge transformation using moment-matching principle
  double w1_new = w1 + w2;
  arma::rowvec w_new(Knew);
  w_new(Knew - 1) = w1_new;
  for(int i=0; i<Knew-1; i++){
    if(i<id1-1){
      w_new(i) = w(i);
    }
    if((i>=id1-1) & (i<id2-1)){
      w_new(i) = w(i+1);
    }
    if(i>=id2-1){
      w_new(i) = w(i+2);
    }
  }
  
  double alpha = w1/w1_new;
  arma::mat Gamma_new(Knew,q);
  gamma_new = (w1 * gamma_1 + w2 * gamma_2)/w1_new;
  Gamma_new = rbind0(removeiii(removeiii(Gamma,id1), id2-1),gamma_new);
  
  arma::rowvec beta(q);
  double ratio_propose = 0;
  for(int i=0; i<q; i++){
    beta(i) = (gamma_new(i) - gamma_1(i)) * std::sqrt(alpha/(1-alpha));
    if((beta(i)<1) & (beta(i)>0)){
      ratio_propose += R::dbeta(beta(i),2,2,true);
    }
  }
  //compute the proposal ratio
  arma::mat temp(1,1);
  if(n1>0){
    for(int i=0; i<n1; i++){
      pro = pro_init;
      ratio_propose += log(pro(0));
    }
  }
  if(n2>0){
    for(int i=0; i<n2; i++){
      pro = pro_init;
      ratio_propose += log(pro(1));
    }
  }
  
  //compute the log Jacobian of moment-matching transformation
  double log_Jacobian = log(w1_new) - (q/2.0) * log(alpha*(1-alpha));
  
  //compute the prior ratio
  double ratio_prior_w = (hyper_delta - 1 + n1) * log(w1) + (hyper_delta - 1 + n2) * log(w2) - 
    (hyper_delta - 1 + nid) * log(w1_new) - log(R::beta(hyper_delta, Knew * hyper_delta));
  arma::mat C_new(Knew, Knew);
  C_new = updateC(Gamma_new,theta,tau);
  double ratio_prior_gamma = log(det(C)) - log(det(C_new));
  double ratio_prior = ratio_prior_w + ratio_prior_gamma;
  double death_ratio = log(K)*2 + ratio_propose  - ratio_prior - log_Jacobian;
  if(Knew ==1){
    death_ratio = death_ratio + log(2);
  }
  
  if(log(R::runif(0,1)) < death_ratio){
    K=Knew;
    Gamma = Gamma_new;
    R = R_new;
    C = C_new;
    w = w_new;
  }
  
  ans["K"] = K;
  ans["w"] = w;
  ans["Gamma"] = Gamma;
  ans["R"]= R;
  ans["C"] = C;
  return (ans);
}

// [[Rcpp::export]]
List RJi_empty(arma::rowvec w,
               int K,
               arma::mat Gamma,
               
               arma::rowvec R,
               
               arma::mat C,
               double theta, double tau,
               int m,  int q,  double hyper_delta = 1){
  
  double merge_or_split = R::runif(0,1);
  if (((K == 1) || (merge_or_split < 0.5)) & (K<10)){
    return (
        Split_empty(w, K, Gamma, R,  C, theta, tau, m,  q,  hyper_delta)
    );
  }
  
  else{
    return (
        Merge_empty(w, K, Gamma, R, C, theta, tau, m,  q,  hyper_delta)
    );
  }
}
