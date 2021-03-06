function f=exgausspdf(mu,sig,tau, x)
% given parameter mu, sig and tau, returns density at x for the ex-Gaussian
% mu, sig, tau are scalars
% x is either a scalar, vector or matrix
% f has the same shape as x
% version 2.0 2/10/99
% (c) Yves Lacouture, Universite Lavale

arg1=(mu./tau)+((sig.*sig)./(2.*tau.*tau))-(x/tau);
arg2=((x-mu)-((sig.*sig)./tau))./sig;
f=(1./tau)*(exp(arg1).*pnf(arg2));


