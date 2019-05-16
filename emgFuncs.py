"""
Exponentially-Modified Gaussian functions for Python

Used  in
*Characterizing the System Impulse Response Function From Photon-Counting LiDAR Data*
by Adam P. Greeley , Thomas A. Neumann, Nathan T. Kurtz, Thorsten Markus, and Anthony J. Martino
to be published in **IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING**.

Digital Object Identifier 10.1109/TGRS.2019.2907230

The regular entry point is **exgausspdf()**.
"""
# April 30, 2019
#

import numpy as np
from scipy.special import erf
from scipy.special import erfc



def pnf(x):
	"""Helper function to compute the cumulative probability for a normalized Gaussian

	:param x: x is scalar or array. Note that x is overwritten by the function evaluation.
	:return: probability with same shape as x

	The pnf is defined as the probability that x<X when X is Gaussian distributed.
	Note that for numerical stability, a different formula is used for
	positive and negative arguments.
	"""
	a = x < 0
	b = x >= 0
	p = x
	m_sqrt2 = np.sqrt(2)
	p[b] = ( (1 + erf(x[b] / m_sqrt2)) / 2)
	p[a] = ( (erfc(-x[a] / m_sqrt2)) / 2)
	return p



def exgausspdf(x,mu,sig,tau):
	"""Compute exponentially-modified Gaussian PDF with parameters mu, sig, and tau.

	:param x: x is scalar or array of values at which to compute the density
	:param mu: scalar: central tendency
	:param sig: scalar: symmetric variablity
	:param tau: scalar: Exponential decay
	:return: probability with same shape as x

	"""
	arg1 = (mu / tau) + ((sig**2) / (2 * (tau**2)) ) - (x / tau)
	arg2 = ((x - mu) - ((sig**2) / tau)) / sig
	f = (1 / tau) * np.exp(arg1) * pnf(arg2)
	return f


