#!/usr/bin/env python3
"""
@author: Daniel N. Blaschke, Dean L. Preston
Date: Mar. 28, 2019 - Feb. 12 2020
updated May 26, 2025: drop python 2.7 support, address some warnings, modernize code
updated Sept. 10, 2025: add missing fitting parameters for Ag, Au, and Ni

Python-implementation of the Hunter-Preston strength model, but with generalized drag coefficient B.
Note: everything in this script is in SI units.

Choose drag coefficient (SI units = Pa s) via commandline option '-Bchoice'.
valid choices are: 'oldconst', 'const', 'linT', 'linear', 'sqrt', 'full'
'const' means constant B, as does 'oldconst' but the latter uses the 'old' version of bminus in the inverse kinetic eqns. and thus reproduces results from Hunter & Preston 2015 for copper.
'linT' means a const times T/300, 'linear' assumes B linear in both T and v (resp. sigma),
and 'sqrt' assumes B linear in T but a 1/sqrt(1-v^2/ct^2) dependence in v which translates to a sqrt(1+A^2sigma^2) dependence in sigma.
Bchoice = 'full' (default) uses the most accurate fit to B for the kinetic eqns., but falls back to 'sqrt' for the inverse ones.
Our paper [Blaschke, Hunter, Preston, Int. J. Plast. 131 (2020) 102750] shows 'const', 'linT' in the top rows and 'sqrt', 'full' in the bottom rows of the side-by-side figs;
'oldconst' is used only in one fig. comparing the inv. kin. eq. to the old model.
Pass option '-hide_inset' to suppress drawing a zoomed-in inset in the kinetic. eq. T-plot (if Bchoice='sqrt' or 'full').

By default, plots are generated for Al and Cu, however other metals may be chosen using the commandline option -metal.
Included in this script are: Ag, Al, Au, Cu, and Ni.
"""
#################################
import argparse
import numpy as np
from scipy.special import erf
from scipy import integrate
from scipy.optimize import fmin
import matplotlib as mpl
mpl.use('Agg', force=False) # don't need X-window, allow running in a remote terminal session
import matplotlib.pyplot as plt
plt.rc('font',**{'family':'serif','size':'11'})
from matplotlib.ticker import AutoMinorLocator
fntsize=11  ## we set our fontsizes below according to this number, so we can change all in one place
## sound speeds, Burgers vectors, and material densities at room temperature:
ct = {'Ag': 1699, 'Al':3109, 'Au': 1183, 'Cu':2322, 'Ni': 2922}
burgers_rt = {'Ag':2.889e-10, 'Al':2.863e-10, 'Au':2.884e-10,'Cu':2.556e-10, 'Ni':2.492e-10}
rho_rt = {'Ag': 10500, 'Al':2700, 'Au': 19300, 'Cu':8960, 'Ni':8900}
#############################
Bchoice = 'full'
# showinset = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-hide_inset', '--hide_inset', action='store_true')
parser.add_argument('-Bchoice','--Bchoice',type=str)
parser.add_argument('-metal','--metal',type=str,default='Al, Cu',help=f"metal / list of metals to plot. default: 'Al, Cu'; available metals: {sorted(list(ct.keys()))}")
args = parser.parse_args()
if args.Bchoice is not None:
    Bchoice = args.Bchoice
showinset = not args.hide_inset

## values of physical constants taken from CRC handbook:
kB = 1.38064852e-23

##########################################
## to include other metals, simply add their numbers in the section below and include their names in this list:
metal = args.metal.replace(" ", "").split(",")
## TODO: implement fits for average B from semi-isotropic calculations

##### model parameters ################
phic = 1/2 # this number follows from Madec et al. 2002 and is applicable to all fcc pure metals (and explicitly checked against experimental data for Ag, Al, Cu by those authors)
tau = 1e-13

#### numbers from Burakovsky-Greef-Preston model for the shear modulus (see shearmodulus_model.py for my implementation of this model):
rho_0 = {'Ag':10630, 'Al':2730, 'Au':19490, 'Cu':9020, 'Ni':8970} ## density at 0K
mu_0 = {'Ag':33.5e9, 'Al':29.3e9, 'Au':30.5e9, 'Cu':52.4e9, 'Ni':93.6e9} ## "cold" shear modulus at 0K and rho_0
T_m_rho0 = {'Ag': 1827, 'Al':1277, 'Au': 2059, 'Cu':1824, 'Ni': 2267} ## melting temperature at cold density rho_0
## model parameters taken from Burakovsky & Preston 2004:
gamma1 = {'Ag':2.23, 'Al':0.84, 'Au':3.21, 'Cu':1.87, 'Ni':1.85} ## units (g/cc)**(1/3)
gamma2 = {'Ag':9.63e4, 'Al':45.4, 'Au':1.97e12, 'Cu':23100, 'Ni':5.60e5} ## units (g/cc)**q
q_exp = {'Ag':4.8, 'Al':3.5, 'Au':9.4, 'Cu':4.7, 'Ni':6.5}
alpha_PW = {'Ag':0.20, 'Al':0.18, 'Au':0.08, 'Cu':0.20, 'Ni':0.37} ## parameter determining mu(rho_0,T) within Preston-Wallace model (1992)

def burg_rho(X,rho):
    '''rescales the Burgers vector according to the present density rho.'''
    return burgers_rt[X]*(rho_rt[X]/rho)**(1/3)

burgers_0 = {} ## Burgers vector magnitude at rho_0 (rounded to same precision as burgers_rt)
for X in metal:
    burgers_0[X] = round(burg_rho(X,rho_0[X]),13)
    
##########################################################

def mu_rho(X,rho):
    '''Computes the shear modulus of metal X as a function of density rho according to the model of Burakovsky, Greeff & Preston 2003.'''
    a2 = 2*gamma2[X]/q_exp[X]
    a3 = 6*gamma1[X]
    ## this fct takes rho in kg/m**3 as input, but model parameters want g/cc, so need to replace all rho by 1e-3*rho below:
    return mu_0[X]*(rho/rho_0[X])**(4/3)*np.exp(-a2*(1/(1e-3*rho)**q_exp[X]-1/(1e-3*rho_0[X])**q_exp[X])-a3*(1/(1e-3*rho)**(1/3)-1/(1e-3*rho_0[X])**(1/3)))

def Tm(X,rho):
    '''returns the melting temperature for metal X as a function of density rho'''
    return T_m_rho0[X]*(rho_0[X]/rho)*mu_rho(X,rho)/mu_0[X]
    
def mu_BGP(X,T,rho):
    '''computes the temperature dependence of mu at constant density rho according to the Preston-Wallace model; input variables are the metal X, temperature T and density rho'''
    return mu_rho(X,rho)*(1-alpha_PW[X]*T/Tm(X,rho))

def ct_rho(X,T,rho):
    '''computes the transverse sound speed as a function of temperature and density (using function mu_BGP() for the shear modulus)'''
    return np.sqrt(mu_BGP(X,T,rho)/rho)
    
######################################################   
print(f"found {Bchoice=}")

def Bratio(X,rho,T=300):
    '''this is a subroutine of B()'''
    out = (rho/rho_rt[X])**(7/6) * np.sqrt(mu_BGP(X,T,rho_rt[X])/mu_BGP(X,T,rho))
    return out

if Bchoice in ('const', 'oldconst'):
    # constant B:
    def B(T=300,rho=1e3,X='Cu'):
        '''returns the drag coefficient B'''
        return 1e-4
elif Bchoice == 'linT':
    # B as simple linear function of T, and density scaling defined by Bscaling above
    def B(T,rho,X):
        '''returns the drag coefficient B'''
        rat = Bratio(X,rho,T)
        out = 5e-5*rat*T/300
        return out
elif Bchoice == 'linear':
    # B as a linear function in T and v (resp. sigma), see PyDislocDyn >=1.2.0 for details and the fits leading to B0 for Bchoice=linear
    # Note: numbers are very sensitive to optimization routine/parameters and changed between original research code and PyDislocDyn 1.2.0 and 1.2.5
    # most of the numbers below were determined with PyDislocDyn 1.3.3 using option --allplots=True; only Al, Cu use older fits to match our IJP paper
    # for the user's convenience, these numbers are re-computed if Bchoice=full and stored in dictionary Boffset[X] (caveat: rounding the fitting parameters leads to small deviations compared to PyDislocDyn)
    B0 = {'Ag': 2.6e-6, 'Al':0.9e-6, 'Au': 2.2e-6, 'Cu':7.1e-6, 'Ni':1.4e-6}
    ### NOTE: at large sigma, T-dep drops out if B(v)=B0(v)*T, because eqn. leads to v(sigma/T); and same for 'sqrt' below
    def B(T,X,sigma,b,rho):
        '''returns the drag coefficient B'''
        rat = Bratio(X,rho,T)
        bct = burg_rho(X,rho)/ct_rho(X,T,rho)
        Bref = B0[X]*rat
        out = Bref*T/300 + sigma*bct ## include approximate density effects neglecting what we don't know: how TOEC and hence the ratio of elastic constants behaves
        return out
elif Bchoice == 'sqrt':
    # B as a linear function in T and a non-linear, albeit simple, function in v (resp. sigma), run this script with Bchoice='full' to recalculate and verify B0 below:
    B0 = {'Ag':5.4e-6, 'Al':4.4e-6, 'Au':15.3e-6, 'Cu':34.3e-6, 'Ni':2.0e-6}
    def B(T,X,sigma,b,rho):
        '''returns the drag coefficient B'''
        rat = Bratio(X,rho,T)
        bct = burg_rho(X,rho)/ct_rho(X,T,rho)
        Bref = B0[X]*rat
        out = Bref*np.sqrt((T/300)**2 + (sigma*bct/Bref)**2) ## include approximate density effects neglecting what we don't know: how TOEC and hence the ratio of elastic constants behaves
        return out
elif Bchoice == 'full':
    def Bedge(v,ct,T,Cfit):
        '''returns the drag coefficient B for pure edge dislocations'''
        c0, c1, c2, c3, c4, c10, c11, c12 = Cfit
        beta = abs(v)/ct ## don't care about the sign of v: any direction is fine
        Tnorm = T/300
        if beta<1:
            out = (c0 - c1*beta + c2*np.log(1-beta**2) + c3*(1/np.sqrt(1-beta**2) - 1) + c4*(1/(1-beta**2)**(3/2) - 1))*Tnorm\
                    + (c10 - c11*beta + c12*(1/(1-beta**2)**(3/2) - 1))*(Tnorm**2-1) ## TODO: include quadratic correction in temperature (no effect if T=300)
            out = max(out,0) ## don't return negative values due to bad fitting parameters
        else: ## if v is equal or larger than ct, set Bedge to +infinity
            out = np.inf
        return 1e-3*out
        
    def Bscrew(v,ct,T,Cfit):
        '''returns the drag coefficient B for pure screw dislocations'''
        c0, c1, c2, c3, c4, c10, c11, c12 = Cfit
        beta = abs(v)/ct ## don't care about the sign of v: any direction is fine
        Tnorm = T/300
        if beta<1:
            out = (c0 - c1*beta + c2*beta**2 + c3*np.log(1-beta**2) + c4*(1/np.sqrt(1-beta**2) - 1))*Tnorm\
                        + (c10 - c11*beta + c12*(1/np.sqrt(1-beta**2) - 1))*(Tnorm**2-1) ## TODO: include quadratic correction in temperature (no effect if T=300)
            out = max(out,0) ## don't return negative values due to bad fitting parameters
        else: ## if v is equal or larger than ct, set Bscrew to +infinity
            out = np.inf
        return 1e-3*out
        
    ## dictionary containing fitting parameters for different metals, different models, different fitting fcts, ...
    ## values determined using PyDislocDyn (github.com/dblaschke-LANL/PyDislocDyn)
    ## tuples are to be read as (c0, c1, c2, c3, c4, c10, c11, c12), see fcts above
    fitting_parameters ={}
    ### including longitudinal phonons and mixed modes, but still room temperature:
    fitting_parameters['Ag'] = {'edge':(0.0079, 0.0051, 0.0018, 0.    , 0.0014, 0, 0, 0), 'screw':(0.0071, 0.0089, 0.0055, 0.0003, 0.0016, 0, 0, 0)}
    fitting_parameters['Al'] = {'edge':(0.0063, 0.0041, 0.0015, 0.0000, 0.0012, 0, 0, 0), 'screw':(0.0057, 0.0069, 0.0043, 0.0003, 0.0014, 0, 0, 0)}
    fitting_parameters['Au'] = {'edge':(0.0246, 0.0241, 0.0295, 0.0602, 0.0005, 0, 0, 0), 'screw':(0.0266, 0.0439, 0.0297, 0.    , 0.0024, 0, 0, 0)}
    fitting_parameters['Cu'] = {'edge':(0.0453, 0.0419, 0.0724, 0.1364, 0.0036, 0, 0, 0), 'screw':(0.0626, 0.0966, 0.0655, 0.0000, 0.0078, 0, 0, 0)}
    fitting_parameters['Ni'] = {'edge':(0.0028, 0.0002, 0.0023, 0.    , 0.0008, 0, 0, 0), 'screw':(0.0015, 0.0011, 0.0016, 0.0007, 0.0011, 0, 0, 0)}

    def Bvel(v,c_t,T,fitedge,fitscrew):
        '''returns the average drag coefficient B=(edge+Bscrew)/2 as a function of velocity v; this function expects fitting parameters in units of mPas and will return B in units of Pas'''
        return (Bedge(v,c_t,T,fitedge) + Bscrew(v,c_t,T,fitscrew))/2
    
    def vr(stress,c_t,T,b,rho,X,fitedge,fitscrew):
        '''returns the velocity if a dislocation in the drag dominated regime as a function of stress; additional input parameters are the transverse sound speed c_t, temperature T, burgers vector b, and fitting parameters to be passed on to Bvel().'''
        bsig = abs(b*stress)
        rat = Bratio(X,rho,T)
        def nonlinear_equation(v):
            return abs(bsig-abs(v)*rat*Bvel(v,c_t,T,fitedge,fitscrew)) ## need abs() if we are to find v that minimizes this expression (and we know that minimum is 0)
        out = fmin(nonlinear_equation,0.01*c_t,disp=False)[0]
        zero = abs(nonlinear_equation(out))
        if zero>1e-5 and zero/bsig>1e-2:
            print(f"Warning: bad convergence for vr({stress=}): eq={zero:.6f}, eq/(b*sig)={zero/bsig:.6f}")
        return out
    
    def B(T,X,sigma,b,rho):
        '''returns drag coefficient B as a function of temperature T, stress sigma and Burgers vector b'''
        fitedge = fitting_parameters[X]['edge']
        fitscrew = fitting_parameters[X]['screw']
        c_t = ct_rho(X,T,rho)
        rat = Bratio(X,rho,T)
        return rat*Bvel(vr(sigma,c_t,T,b,rho,X,fitedge,fitscrew),c_t,T,fitedge,fitscrew)
        
    ## fall back to 'sqrt' for the inverse kinetic eqns.
    print("Warning: using Bchoice='full' only for the kinetic eqns., falling back to Bchoice='sqrt' for the inverse ones!")
    ## recalculate B0:
    Bmin = {}
    B0 = {}
    Boffset = {}
    for X,params in fitting_parameters.items():
        c_t = ct_rho(X,300,rho_rt[X])
        Bmin[X] = np.zeros((1000)) ## find minimum value
        velocities = np.linspace(0,0.8*c_t,len(Bmin[X]))
        for i in range(len(Bmin[X])):
            Bmin[X][i] = Bvel(velocities[i],c_t,300,params['edge'],params['screw'])
        Bmin[X] = np.min(Bmin[X])
        B0[X] = round((B(300,X,0,burgers_rt[X],rho_rt[X])+3*Bmin[X])/4,7)
        ## code below determines Boffset[X], which is used as B0 in Bchoice=linear
        burg = burgers_rt[X]
        ### compute what stress is needed to move dislocations at velocity v:
        def sigma_eff(v):
            return v*Bvel(v,c_t,300,params['edge'],params['screw'])/burg
        ## B as a straight line:
        def Bstraight(sigma,Boffset=0):
            return Boffset+sigma*burg/c_t
        sigma_max = min(1.5e9,float(sigma_eff(0.99*c_t)))
        Boffset[X] = round(float(B(300,X,sigma_max,burg,rho_rt[X])-Bstraight(sigma_max,0)),7)
else:
    raise ValueError(f"{Bchoice=} not implemented")
    
######################################################

### note: density of (im)mobile dislocations rhoi/rhom must also be given in SI units, i.e. 1/m^2 (hence 4 orders of magnitude larger numbers than the 1/cm^2 that the mmtk notebook wants)
######################################################
def tension(X, rhoi, rho, T):
    '''computes the dislocation line tension'''
    b = burg_rho(X,rho)
    return -(5/16/np.pi)*mu_BGP(X,T,rho)*b**2*np.log(b*np.sqrt(rhoi))

######################################################
    
def sigmac(X, rhoi, rho, T):
    '''computes the critical stress'''
    b = burg_rho(X,rho)
    return 2*phic*np.sqrt(rhoi)*tension(X, rhoi, rho, T)/b
    
def sigmab(X, rhoi, rho, T):
    '''computes the back stress'''
    b = burg_rho(X,rho)
    gb = 1/5
    return gb*mu_BGP(X,T,rho)*b*np.sqrt(rhoi)

def heaviside(x):
    '''identical to numpy's step function heaviside(x,1/2)'''
    return np.heaviside(x,1/2)
        
def sigma(X, sigmaa, rhoi, rho, T):
    '''computes the effective (local) stress'''
    out = sigmaa - sigmab(X,rhoi,rho,T)
    return out*heaviside(out)

if Bchoice in ('const', 'linT', 'oldconst'):
    def tB(X, rhoi, rho, T):
        '''computes the bowout time'''
        return B(T,rho,X)/(rhoi*np.pi**2*tension(X, rhoi, rho, T))
elif Bchoice in ('linear', 'sqrt', 'full'):
    ## in this case, tB is defined as tB-tilde
    def tB(X, rhoi, rho, T):
        '''computes the bowout time'''
        rat = Bratio(X,rho,T)
        Bref = B0[X]*rat
        return Bref/(rhoi*np.pi**2*tension(X, rhoi, rho, T))
        
phicpi = 2*phic/np.pi**2 + 1

def phimax(X,sigmaa, rhoi, rho, T):
    '''computes the maximum bowout angle'''
    return -(sigma(X,sigmaa, rhoi, rho, T)/sigmac(X,rhoi, rho, T))*phic
    
def EE(X,rhoi, rho, T):
    '''computes the activation energy at zero stress'''
    kappa = 1
    b = burg_rho(X,rho)
    return kappa*b*phic**2*tension(X, rhoi, rho, T)

def Afct(X,rhoi, rho, T):
    '''computes parameter A as a function of activation energy and temperature'''
    return np.sqrt(EE(X,rhoi, rho, T)/(kB*T))

#### kinetic eqns:

### integrate.quad options (to trade-off accuracy for speed in the kinetic eqns.)
epsabs=1.49e-04 ## absolute error tolerance; default: 1.49e-08
epsrel=1.49e-04 ## relative error tolerance; default: 1.49e-08
limit=30 ## max no of subintervals; default: 50
###
  
if Bchoice in ('const', 'linT', 'oldconst'):
    def tW(X,sigmaa, rhoi, rho, T, sig_small=0.35, sig_large=100):
        '''computes the wait time'''
        A = Afct(X, rhoi, rho, T)
        sig = sigma(X,sigmaa, rhoi, rho, T) / sigmac(X,rhoi, rho, T)
        tbow = tB(X,rhoi, rho, T)
        if sig_small is None: ## determine automatically within range [0.35,0.8] if not set explicitly
            sig_small = min(0.8,max(0.35,0.055*A)) ## need to transition sooner for large A due to convergence issues in integral over erf()
        if sig < sig_small:
            ## approximation for low stress (not only saves computation time, but also circumvents convergence problems with this integrals at low stress
            ## derive by approximating sig*(1 - np.exp(-z)) ~> sig inside f(x,sig,A), then integrating analytically
            # out = (1 + erf(A))*tau/(1 - erf(A*(1 - sig))) ## problem: large A and small sig lead to "divide by zero errors", better use asymptotic expansion of erf():
            Asig = A*(1 - sig)
            out = (1 + erf(A))*tau*np.sqrt(np.pi)*np.exp(Asig**2)*(Asig+1/(2*Asig))
            # out = 2*tau*np.sqrt(np.pi)*Asig*np.exp(Asig**2) ## approximation from the paper
            ##
            Asig_m = A*(1 + sig)
            if Asig_m<25: ## avoid overflow in exp(), tmin tends to inf at such high Asig_m anyway and hence has no effect in that case
                tmin = (1 + erf(A))*tau*np.sqrt(np.pi)*(Asig_m+1/(2*Asig_m))*np.exp(Asig_m**2)
                # tmin = 2*tau*np.sqrt(np.pi)*Asig_m*np.exp(Asig_m**2) ## has neglegible effect unless sig is very small (i.e. tmin>>out otherwise)
                out = 1/(1/out - 1/tmin) ## accounting for reverse glide, see eqn. (89) in the paper
            ##
        elif sig>sig_large:
            ## approximation for very high stress (see paper for derivation); this is a rough estimate for tW, but unimportant in the drag dominated regime
            out = tbow/sig + tau
        else:
            def f(x, sig, A):
                return integrate.quad(lambda z: (1/2)*(1 - erf(A*(1 - sig*(1 - np.exp(-z))))), 0, x, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
            out = (1/2)*(1 + erf(A))*tbow*integrate.quad(lambda x: np.exp(-(tbow/tau)*f(x, sig, A)), 0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        return out
        
    def tT0(X, sigmaa, rhoi, rho, T):
        '''computes the run time between obstacles'''
        b = burg_rho(X,rho)
        return B(T,rho,X)/(b*np.sqrt(rhoi)*sigma(X,sigmaa, rhoi, rho, T))
    
    def strainrate(X, logsighat, rhom, rhoi, rho, T):
        '''computes the strain rate as a function of stress (among other parameters)'''
        sigm = (10**logsighat)*sigmac(X,rhoi, rho, T)
        sigmaa = sigm + sigmab(X,rhoi, rho, T)
        b = burg_rho(X,rho)
        tWa = tW(X,sigmaa, rhoi, rho, T)
        if tWa<0:
            tWa = tW(X,sigmaa, rhoi, rho, T, sig_small=0.8)
        zeta = 1/(1 - 0.25*np.exp(-np.log10(tWa/tT0(X,sigmaa, rhoi, rho, T))**2))
        out = b*rhom*zeta/(np.sqrt(rhoi)*tWa + B(T,rho,X)/(b*sigm))
        return np.log10(out)
     
elif Bchoice in ('linear', 'sqrt', 'full'):
    ## since execution of B() is slow for 'full', save some time by allowing strainrate() to pass Bdrag to subroutines tW() and tT0()
    def tW(X,sigmaa, rhoi, rho, T, Bdrag=None, sig_small=None, sig_large=100):
        '''computes the wait time'''
        A = Afct(X, rhoi, rho, T)
        sigm = sigma(X,sigmaa, rhoi, rho, T)
        sig = sigm / sigmac(X,rhoi, rho, T)
        if sig_small is None: ## determine automatically within range [0.35,0.8] if not set explicitly
            sig_small = min(0.8,max(0.35,0.055*A)) ## need to transition sooner for large A due to convergence issues in integral over erf()
        if sig < sig_small:
            ## approximation for low stress (not only much faster, but also circumvents convergence problems with these integrals at low stress
            ## derive by approximating sig*(1 - np.exp(-z)) ~> sig inside f(x,sig,A), then integrating analytically
            # out = (1 + erf(A))*tau/(1 - erf(A*(1 - sig))) ## problem: large A and small sig lead to "divide by zero errors", better use asymptotic expansion of erf():
            Asig = A*(1 - sig)
            out = (1 + erf(A))*tau*np.sqrt(np.pi)*np.exp(Asig**2)*(Asig+1/(2*Asig))
            # out = 2*tau*np.sqrt(np.pi)*Asig*np.exp(Asig**2) ## approximation from the paper
            ##
            Asig_m = A*(1 + sig)
            if Asig_m<25: ## avoid overflow in exp(), tmin tends to inf at such high Asig_m anyway and hence has no effect in that case
                tmin = (1 + erf(A))*tau*np.sqrt(np.pi)*(Asig_m+1/(2*Asig_m))*np.exp(Asig_m**2)
                # tmin = 2*tau*np.sqrt(np.pi)*Asig_m*np.exp(Asig_m**2) ## has neglegible effect unless sig is very small (i.e. tmin>>out otherwise)
                out = 1/(1/out - 1/tmin) ## accounting for reverse glide, see eqn. (89) in the paper
            ##
        elif sig>sig_large:
            if Bdrag is None:
                b = burg_rho(X,rho)
                Bdrag=B(T,X,sigm,b,rho)
            tbow = Bdrag/(rhoi*np.pi**2*tension(X, rhoi, rho, T))
            ## approximation for very high stress (see paper for derivation); this is a rough estimate for tW, but unimportant in the drag dominated regime
            out = tbow/sig + tau
        else:
            if Bdrag is None:
                b = burg_rho(X,rho)
                Bdrag=B(T,X,sigm,b,rho)
            tbow = Bdrag/(rhoi*np.pi**2*tension(X, rhoi, rho, T))
            def f(x, sig, A):
                return integrate.quad(lambda z: (1/2)*(1 - erf(A*(1 - sig*(1 - np.exp(-z))))), 0, x, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
            out = (1/2)*(1 + erf(A))*tbow*integrate.quad(lambda x: np.exp(-(tbow/tau)*f(x, sig, A)), 0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        return out
        
    def tT0(X, sigmaa, rhoi, rho, T, Bdrag=None):
        '''computes the run time between obstacles'''
        b = burg_rho(X,rho)
        if Bdrag is None:
            Bdrag=B(T,X,sigma(X,sigmaa, rhoi, rho, T),b,rho)
        return Bdrag/(b*np.sqrt(rhoi)*sigma(X,sigmaa, rhoi, rho, T))
    
    def strainrate(X, logsighat, rhom, rhoi, rho, T):
        '''computes the strain rate as a function of stress (among other parameters)'''
        sigm = (10**logsighat)*sigmac(X,rhoi, rho, T)
        sigmaa = sigm + sigmab(X,rhoi, rho, T)
        b = burg_rho(X,rho)
        Bdrag=B(T,X,sigm,b,rho)
        tWa = tW(X,sigmaa, rhoi, rho, T, Bdrag)
        if tWa<0:
            ### if rho is large, the double integral within tWa in the low intermediate stress regime may fail leading to a negative value, in this case fall back to small stress approximation (which is accurate enough in this case)
            tWa = tW(X,sigmaa, rhoi, rho, T, Bdrag,sig_small=0.8)
        zeta = 1/(1 - 0.25*np.exp(-np.log10(tWa/tT0(X,sigmaa, rhoi, rho, T, Bdrag))**2))
        return np.log10(b*rhom*zeta/(np.sqrt(rhoi)*tWa + Bdrag/(b*sigm)))
    
#### inverse kinetic eqns:

lg10e = np.log10(np.e)
## mminus is a direct consequence of the approximation eqn. (99) in the paper, i.e. the choice for the slope of the linear approximation to h(y) ~ -A^2y-const.
## as such, it is somewhat arbitrary ...
## In fact, the asymptotic behavior for h(y) as y->-inf is a constant, A^2, meaning zero slope,
## but if reverse glide is taken into account (see eqn. (89) in the paper), this changes to ~ -y*ln(10)=-y/log10(e) as y->-inf
## using again a linear approximation, but with this new slope leads to mminus=1 and dependening on the chosen constant, a A-dependent shift of bminus
## problems with this latter approximation: not good in the regime -1~<y<0 because it relied on the expansion for y->-inf
## worse yet: with this new slope mplus-mminus=0 and xc=+inf, so cannot interpolate
def mminus(A):
    '''this subroutine returns parameter m-minus'''
    return 1/(A**2*lg10e)
    
def bminus_old(X, A, rhom, rhoi, rho):
    '''this subroutine returns parameter bminus of the original Hunter-Preston model'''
    b = burg_rho(X,rho)
    return -mminus(A)*np.log10((b*rhom/(tau*np.sqrt(rhoi))))

mplus = 1

if Bchoice in ('const', 'linT'):
    def bplus(X, rhom, rhoi, rho, T, x):
        '''this subroutine returns parameter bplus'''
        b = burg_rho(X,rho)
        return np.log10((phicpi/(b*rhom))*B(T,rho,X)/(b*sigmac(X, rhoi, rho, T)))
    def bminus(X, A, rhom, rhoi, rho, T):
        '''this subroutine returns parameter bminus'''
        b = burg_rho(X,rho)
        ### additional term is neglegible for low to moderate temperatures, but quickly grows for high temperatures where its effect is such that xc is moved to negative number
        return -mminus(A)*np.log10((b*rhom/(tau*np.sqrt(rhoi)))) - np.log10(1+A*B(T,rho,X)/(b*sigmac(X,rhoi,rho,T)*np.sqrt(rhoi)*tau)/np.exp(A**2))*(mplus-mminus(A))
    def xmax(X, rhom, rho, T, rhoi):
        ### there is no maximum strain rate in the inverse kinetic equations due to neglecting tau in these cases
        ### but there the limit tW->tB*sigc/sighat + tau -> tau as sighat-> inf does imply a maximum strain rate for the kinetic equations, which is what we calculate here
        # return np.inf
        '''computes the maximum strain rate due to finite time scale tau'''
        return np.log10(rhom*burg_rho(X,rho)/(np.sqrt(rhoi)*tau))
elif Bchoice=='oldconst':
    def bplus(X, rhom, rhoi, rho, T, x):
        '''this subroutine returns parameter bplus'''
        b = burg_rho(X,rho)
        return np.log10((phicpi/(b*rhom))*B(T,rho,X)/(b*sigmac(X, rhoi, rho, T)))
    def bminus(X, A, rhom, rhoi, rho, T):
        '''this subroutine returns parameter bminus'''
        return bminus_old(X, A, rhom, rhoi, rho)
    def xmax(X, rhom, rho, T, rhoi):
        '''returns infinity in order to ignore the maximum strain rate in this case'''
        return np.inf
        # return np.log10(rhom*burg_rho(X,rho)/(np.sqrt(rhoi)*tau))
elif Bchoice == 'linear':
    def bplus(X, rhom, rhoi, rho, T,x):
        '''this subroutine returns parameter bplus'''
        rat = Bratio(X,rho,T)
        Bref = B0[X]*rat
        b = burg_rho(X,rho)
        c_t = ct_rho(X,T,rho)
        sigc = sigmac(X, rhoi, rho, T)
        brak = phicpi*Bref/(b*sigc)
        out = np.log10((1/(b*rhom))*brak*T/300)
        return out - np.log10(1 - brak*sigc*10**x/(Bref*c_t*rhom))
    def bminus(X, A, rhom, rhoi, rho, T):
        '''this subroutine returns parameter bminus'''
        rat = Bratio(X,rho,T)
        Bref = B0[X]*rat
        b = burg_rho(X,rho)
        ### additional term is neglegible for low to moderate temperatures, but quickly grows for high temperatures where its effect is such that xc is moved to negative number
        return -mminus(A)*np.log10((b*rhom/(tau*np.sqrt(rhoi)))) - np.log10(1+A*Bref/(b*sigmac(X,rhoi,rho,T)*np.sqrt(rhoi)*tau)/np.exp(A**2))*(mplus-mminus(A))
    def xmax(X, rhom, rho, T, rhoi=1e12):
        '''computes the maximum strain rate due to diverging drag coefficient B at the transverse sound speed'''
        b = burg_rho(X,rho)
        c_t = ct_rho(X,T,rho)
        return np.log10((b*c_t*rhom)/(phicpi))
    
elif Bchoice in ('sqrt', 'full'):
    ### note: diverging B(v) leads to B(sigma) linear in sigma as sigma->infty, thereby introducing a maximum strain rate of epsilon_max=b*rhom*ct/phicpi
    def bplus(X, rhom, rhoi, rho, T,x):
        '''this subroutine returns parameter bplus'''
        rat = Bratio(X,rho,T)
        Bref = B0[X]*rat
        b = burg_rho(X,rho)
        c_t = ct_rho(X,T,rho)
        sigc = sigmac(X, rhoi, rho, T)
        brak = phicpi*Bref/(b*sigc)
        out = np.log10((1/(b*rhom))*brak*T/300)
        return out - 0.5*np.log10(1 - (brak*sigc*10**x/(Bref*c_t*rhom))**2)
    def bminus(X, A, rhom, rhoi, rho, T):
        '''this subroutine returns parameter bminus'''
        rat = Bratio(X,rho,T)
        Bref = B0[X]*rat
        b = burg_rho(X,rho)
        ### additional term is neglegible for low to moderate temperatures, but quickly grows for high temperatures where its effect is such that xc is moved to negative number
        return -mminus(A)*np.log10((b*rhom/(tau*np.sqrt(rhoi)))) - np.log10(1+A*Bref/(b*sigmac(X,rhoi,rho,T)*np.sqrt(rhoi)*tau)/np.exp(A**2))*(mplus-mminus(A))
    def xmax(X, rhom, rho, T, rhoi=1e12):
        '''computes the maximum strain rate due to diverging drag coefficient B at the transverse sound speed'''
        b = burg_rho(X,rho)
        c_t = ct_rho(X,T,rho)
        return np.log10((b*c_t*rhom)/(phicpi))

def xc(X, A, rhom, rhoi, rho, T, x):
    '''computes the interpolation point'''
    return (bminus(X, A, rhom, rhoi, rho, T) - bplus(X, rhom, rhoi, rho, T, x))/(mplus - mminus(A))
    
def logsig(X, rhom, rhoi, rho, T, x, dx):
    '''computes stress as a function of strain rate (among other parameters)'''
    A = Afct(X, rhoi, rho, T)
    m_min = mminus(A)
    b_min = bminus(X, A, rhom, rhoi, rho,T)
    b_pl = bplus(X, rhom, rhoi, rho, T, x)
    x_c = (b_min - b_pl)/(mplus - m_min)
    return 0.5*((mplus + m_min)*x + b_pl + b_min) + (mplus - m_min)*dx/2*np.log(2*np.cosh((x - x_c)/dx))
                   
#########
if __name__ == '__main__':
        
    ########### make some nice plots:
    if Bchoice == 'full':
        ### start with BofSigma for all metals:
        resolution = 300
        sigscaled = np.linspace(0,4,resolution)
        Bfit = {}
        for X in metal:
            sig0 = ct_rho(X,300,rho_rt[X])*B0[X]/burgers_rt[X]
            Bfit[X] = np.zeros((resolution))
            for i in range(resolution):
                Bfit[X][i] = B(300,X,sig0*sigscaled[i],burgers_rt[X],rho_rt[X])/B0[X]
            
        fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,3.5))
        ax.set_xlabel(r'$\sigma b/(c_\mathrm{t}B_0)$',fontsize=fntsize)
        ax.set_ylabel(r'$B/B_0$',fontsize=fntsize)
        # ax.set_title("",fontsize=fntsize)
        ax.axis((0,4,0,4))
        sig4 = np.linspace(0,4,len(Bfit[X]))
        ax.plot(sig4,np.sqrt(1+sig4**2),':',color='gray',label=r"$\sqrt{1+\left(\frac{\sigma b}{c_\mathrm{t}B_0}\right)^2}$")
        for X in metal:
            ax.plot(sigscaled,Bfit[X],label=X)
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='best',handlelength=1.1, frameon=False, shadow=False,fontsize=fntsize)
        plt.savefig("B_of_sigma_normalized.pdf",format='pdf',bbox_inches='tight')
        plt.close()
    
    for X in metal:
        print(f"'{X}': plotting inverse kinetic eqns.\n")
        
        ### INVERSE KINETIC EQNS: 
        resolution = 1000 ## choose resolution for the plots (i.e. number of points to compute)
        ## choose range of log10(strain rates) for x-axis
        x_min = -3
        x_max = 10
        epsdot = np.linspace(x_min,x_max,resolution)
        sig = np.zeros(resolution)
        
        ## compare rhoi
        fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
        ax.set_xlabel(r'$\mathrm{log}_{10}(\dot\epsilon_\mathrm{p}/\mathrm{s}^{-1})$',fontsize=fntsize)
        ax.set_ylabel(r'$\mathrm{log}_{10}(\hat\sigma/\sigma_c)$',fontsize=fntsize)
        xm1 = xmax(X,1e12,rho_0[X],T=300,rhoi=1e16)
        if xm1>x_max:
            epsdot_short = np.copy(epsdot)
            xm3 = x_max-1
        else:
            epsdot_short = np.concatenate((np.linspace(x_min,0.99*xm1,resolution),np.linspace(0.99*xm1,xm1-1e-15,resolution)))
            xm3 = xm1
        epsdotlen = len(epsdot_short)
        ax.axis((x_min,int(xm3+1),-2,3))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(f"{X}",fontsize=fntsize)
        for rhi in [1e10,1e12,1e14,1e16]:
            sig = np.zeros(epsdotlen) ## reset
            for x in range(epsdotlen):
                with np.errstate(invalid='ignore'):
                    sig[x] = logsig(X, 1e12, rhi, rho_0[X], 300, epsdot_short[x], 0.75)
            ax.plot(epsdot_short,sig,label=r'$\rho_i=10^{'+f"{int(np.log10(rhi))}"+r'}$m$^{-2}$')
        ax.legend(loc='best',fontsize=fntsize)
        plt.savefig(f"inversekinetic_{X}_rhoi.pdf",format='pdf',bbox_inches='tight')
        plt.close()
        
        ## compare rhom
        fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
        ax.set_xlabel(r'$\mathrm{log}_{10}(\dot\epsilon_\mathrm{p}/\mathrm{s}^{-1})$',fontsize=fntsize)
        ax.set_ylabel(r'$\mathrm{log}_{10}(\hat\sigma/\sigma_c)$',fontsize=fntsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(f"{X}",fontsize=fntsize)
        xm1 = xmax(X,1e10,rho_0[X],T=300,rhoi=1e12)
        xm2a = xmax(X,1e12,rho_0[X],T=300,rhoi=1e12)
        xm2b = xmax(X,1e14,rho_0[X],T=300,rhoi=1e12)
        xm3 = xmax(X,1e16,rho_0[X],T=300,rhoi=1e12)
        if max(xm1,xm2a,xm2b,xm3)>x_max:
            epsdot_short = np.copy(epsdot)
            xm3 = x_max-1
        else:
            epsdot_short = np.concatenate((np.linspace(x_min,xm1-1e-15,resolution),np.linspace(xm1,xm2a-1e-15,resolution),np.linspace(xm2a,xm2b-1e-15,resolution),np.linspace(xm2b,xm3,resolution)))
        epsdotlen = len(epsdot_short)
        y_min = int(2*logsig(X, 1e16, 1e12, rho_0[X], 300, x_min, 0.75)-1)/2
        ax.axis((x_min,int(xm3+1),y_min,3))
        for rhm in [1e10,1e12,1e14,1e16]:
            sig = np.zeros(epsdotlen) ## reset
            for x in range(epsdotlen):
                with np.errstate(invalid='ignore'):
                    sig[x] = logsig(X, rhm, 1e12, rho_0[X], 300, epsdot_short[x], 0.75)
            ax.plot(epsdot_short,sig,label=r'$\rho_m=10^{'+f"{int(np.log10(rhm))}"+r'}$m$^{-2}$')
        ax.legend(loc='best',fontsize=fntsize)
        plt.savefig(f"inversekinetic_{X}_rhom.pdf",format='pdf',bbox_inches='tight')
        plt.close()
        
        ## compare T
        fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
        ax.set_xlabel(r'$\mathrm{log}_{10}(\dot\epsilon_\mathrm{p}/\mathrm{s}^{-1})$',fontsize=fntsize)
        ax.set_ylabel(r'$\mathrm{log}_{10}(\hat\sigma/\sigma_c)$',fontsize=fntsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(f"{X}",fontsize=fntsize)
        xm1 = xmax(X,1e12,rho_0[X],T=0.95*T_m_rho0[X],rhoi=1e12)
        xm2 = xmax(X,1e12,rho_0[X],T=T_m_rho0[X]/2,rhoi=1e12)
        xm3 = xmax(X,1e12,rho_0[X],T=300,rhoi=1e12)
        if xm1>x_max:
            epsdot_short = np.copy(epsdot)
            xm3 = x_max-1
        else:
            epsdot_short = np.concatenate((np.linspace(x_min,0.99*xm1,resolution),np.linspace(0.99*xm1,xm1-1e-15,resolution),np.linspace(xm1,xm2-1e-15,resolution),np.linspace(xm2,xm3,resolution)))
        epsdotlen = len(epsdot_short)
        ax.axis((x_min,int(xm3+1),-5,4))
        for T in [300,T_m_rho0[X]/3,T_m_rho0[X]/2,0.95*T_m_rho0[X]]:
            sig = np.zeros(epsdotlen) ## reset
            for x in range(epsdotlen):
                with np.errstate(invalid='ignore'):
                    sig[x] = logsig(X, 1e12, 1e12, rho_0[X], T, epsdot_short[x], 0.75)
            ax.plot(epsdot_short,sig,label=r'$T=$'+f"{T:.0f}"+r'K')
        # ax.plot(epsdot_short,-np.ones(epsdotlen),'k:')
        ax.legend(loc='best',fontsize=fntsize)
        # ax.legend(loc='upper left',ncol=2,fontsize=fntsize)
        plt.savefig(f"inversekinetic_{X}_T.pdf",format='pdf',bbox_inches='tight')
        plt.close()
        
        ## compare rho
        fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
        ax.set_xlabel(r'$\mathrm{log}_{10}(\dot\epsilon_\mathrm{p}/\mathrm{s}^{-1})$',fontsize=fntsize)
        ax.set_ylabel(r'$\mathrm{log}_{10}(\hat\sigma/\sigma_c)$',fontsize=fntsize)
        xm1 = xmax(X,1e12,rho_0[X],T=300,rhoi=1e12)
        xm2 = xmax(X,1e12,1.5*rho_0[X],T=300,rhoi=1e12)
        xm3 = xmax(X,1e12,2*rho_0[X],T=300,rhoi=1e12)
        if xm1>x_max:
            epsdot_short = np.copy(epsdot)
            xm3 = x_max-1
        else:
            epsdot_short = np.concatenate((np.linspace(x_min,0.99*xm1,resolution),np.linspace(0.99*xm1,xm1-1e-15,resolution),np.linspace(xm1,xm2-1e-15,resolution),np.linspace(xm2,xm3,resolution)))
        epsdotlen = len(epsdot_short)
        ax.axis((x_min,int(xm3+1),-1,3))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(f"{X}",fontsize=fntsize)
        for rh in [rho_0[X],1.5*rho_0[X],2*rho_0[X]]:
            sig = np.zeros(epsdotlen) ## reset
            for x in range(epsdotlen):
                with np.errstate(invalid='ignore'):
                    sig[x] = logsig(X, 1e12, 1e12, rh, 300, epsdot_short[x], 0.75)
            ax.plot(epsdot_short,sig,label=r'$\rho='+f"{rh/rho_0[X]:.1f}"+r'\rho_0$')
        ax.legend(loc='best',fontsize=fntsize)
        plt.savefig(f"inversekinetic_{X}_rho0.pdf",format='pdf',bbox_inches='tight')
        plt.close()
        
        #### compare old and new bminus as a fct of A
        fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
        ax.set_xlabel(r'$A(T)$',fontsize=fntsize)
        ax.set_ylabel(r'$b_-$',fontsize=fntsize)
        T = np.linspace(300,0.95*T_m_rho0[X],resolution)
        Npts = len(T)
        A = np.zeros((Npts))
        ax.axis((Afct(X,1e12,rho_0[X],T[-1]),Afct(X,1e12,rho_0[X],T[0]),-6,0))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(f"{X}",fontsize=fntsize)
        bm_new = np.zeros((Npts))
        bm_old = np.zeros((Npts))
        for x in range(Npts):
            A[x] = Afct(X,1e12,rho_0[X],T[x])
            bm_new[x] = bminus(X,A[x],1e12,1e12,rho_0[X],T[x])
            bm_old[x] = bminus_old(X,A[x],1e12,1e12,rho_0[X])
        ax.plot(A,bm_new,label=r'$b_-^\mathrm{new}$')
        ax.plot(A,bm_old,label=r'$b_-^\mathrm{old}$')
        ax.legend(loc='lower right',fontsize=fntsize)
        ax2 = ax.twiny()
        Tlabels = np.round(np.array([300,T_m_rho0[X]/3,T_m_rho0[X]/2,0.95*T_m_rho0[X]]),0).astype(int)
        def AT(T):
            '''this is a subroutine of the plotting routine for bminus'''
            return Afct(X,1e12,rho_0[X],T)
        Tpos = [AT(x) for x in Tlabels]
        ax2.set_xticks(Tpos)
        ax2.set_xticklabels(Tlabels)
        # ax2.set_xticklabels([r"300K", r"$T_m/3$", r"$T_m/2$", r"$0.95T_m$"])
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 36))
        ax2.set_xlabel(r'$T$[K]')
        # ax2.set_xlabel(r'$T$')
        ax2.set_xlim(ax.get_xlim())
        plt.savefig(f"bminus_{X}.pdf",format='pdf',bbox_inches='tight')
        plt.close()
        
        
        ### KINETIC EQNS:
        plotrhoi = True ## set any of these to False to bypass that plot
        plotrhom = True
        plotT = True
        plotrho = True
        ###
        if Bchoice in ('oldconst', 'const', 'linT'):
            showinset=False ## nothing to show in those cases
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        ## choose range of log10(stress/stress_crit) for x-axis:
        siga_res = 500
        sigx_min = -2
        sigx_max = 4
        ## use half of the points for [sigx_min,0) and the other half for [0,sigx_max]
        siga = np.concatenate((np.linspace(sigx_min,0,int(siga_res/2),endpoint=False),np.linspace(0,sigx_max,int(siga_res/2))))
        # sigratio = 10**siga
        straindot = np.zeros(siga_res)
        
        if plotrhoi:
            ## compare rhoi
            print("plotting kinetic eqns: rhoi")
            fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
            ax.set_ylabel(r'$\mathrm{log}_{10}(\dot\epsilon_\mathrm{p}/\mathrm{s}^{-1})$',fontsize=fntsize)
            ax.set_xlabel(r'$\mathrm{log}_{10}(\hat\sigma/\sigma_c)$',fontsize=fntsize)
            new_xmax=int(min(x_max,xmax(X,1e12,rho_0[X],T=300,rhoi=1e16)+1)) # stop plot at next higher integer past log10 of the maximum strain rate (if any) at given rhom
            ax.axis((sigx_min,sigx_max,x_min,new_xmax))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_title(f"{X}",fontsize=fntsize)
            for rhi in [1e10,1e12,1e14,1e16]:
                straindot = np.zeros(siga_res) ## reset
                for x,sigax in enumerate(siga):
                    straindot[x] = strainrate(X, sigax, 1e12, rhi, rho_0[X], 300)
                ax.plot(siga,straindot,label=r'$\rho_i=10^{'+f"{int(np.log10(rhi))}"+r'}$m$^{-2}$')
                # print("#pts with low-stress approx: {}, with high-stress approx: {}, A={:.4f}".format(len(sigratio[sigratio<0.35]),len(sigratio[sigratio>100]),Afct(X, rhi, rho_0[X], 300)))
            ax.legend(loc='best',fontsize=fntsize)
            plt.savefig(f"kinetic_{X}_rhoi.pdf",format='pdf',bbox_inches='tight')
            plt.close()
            
        if plotrhom:
            ## compare rhom
            print("rhom")
            fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
            ax.set_ylabel(r'$\mathrm{log}_{10}(\dot\epsilon_\mathrm{p}/\mathrm{s}^{-1})$',fontsize=fntsize)
            ax.set_xlabel(r'$\mathrm{log}_{10}(\hat\sigma/\sigma_c)$',fontsize=fntsize)
            ax.axis((sigx_min,sigx_max,x_min,x_max))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_title(f"{X}",fontsize=fntsize)
            for rhm in [1e10,1e12,1e14,1e16]:
                straindot = np.zeros(siga_res) ## reset
                for x,sigax in enumerate(siga):
                    straindot[x] = strainrate(X, sigax, rhm, 1e12, rho_0[X], 300)
                ax.plot(siga,straindot,label=r'$\rho_m=10^{'+f"{int(np.log10(rhm))}"+r'}$m$^{-2}$')
                # print("#pts with low-stress approx: {}, with high-stress approx: {}, A={:.4f}".format(len(sigratio[sigratio<0.35]),len(sigratio[sigratio>100]),Afct(X, 1e12, rho_0[X], 300)))
            ax.legend(loc='best',fontsize=fntsize)
            plt.savefig(f"kinetic_{X}_rhom.pdf",format='pdf',bbox_inches='tight')
            plt.close()
            
        if plotT:
            ## compare T
            print("T")
            fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
            ax.set_ylabel(r'$\mathrm{log}_{10}(\dot\epsilon_\mathrm{p}/\mathrm{s}^{-1})$',fontsize=fntsize)
            ax.set_xlabel(r'$\mathrm{log}_{10}(\hat\sigma/\sigma_c)$',fontsize=fntsize)
            new_xmax=int(min(x_max,strainrate(X, sigx_max, 1e12, 1e12, rho_0[X], 300)+1,xmax(X,1e12,rho_0[X],T=300,rhoi=1e12)+1)) # stop plot at next higher integer past log10 of the maximum strain rate (if any) at given rhom
            ax.axis((sigx_min,sigx_max,x_min,new_xmax))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_title(f"{X}",fontsize=fntsize)
            T_vals = [300,T_m_rho0[X]/3,T_m_rho0[X]/2,0.95*T_m_rho0[X]]
            straindot = np.zeros((len(T_vals),siga_res)) ## reset
            for Ti,T in enumerate(T_vals):
                for x,sigax in enumerate(siga):
                    straindot[Ti,x] = strainrate(X, sigax, 1e12, 1e12, rho_0[X], T)
                ax.plot(siga,straindot[Ti],label=r'$T=$'+f"{T:.0f}"+r'K')
                # print("#pts with low-stress approx: {}, with high-stress approx: {}, A={:.4f}".format(len(sigratio[sigratio<0.35]),len(sigratio[sigratio>100]),Afct(X, 1e12, rho_0[X], T)))
            #### inset:
            if showinset:
                axins = zoomed_inset_axes(ax, 2.5, loc=10, bbox_to_anchor=(0.1,0.1,1,1), bbox_transform=ax.transAxes) ## loc=10 ='center'
                for Ti,T in enumerate(T_vals):
                    axins.plot(siga,straindot[Ti],label=r'$T=$'+f"{T:.0f}"+r'K',linewidth=2.0)
                if X=='Al':
                    x1, x2, y1, y2 = 0.7, 1.6, 4.9, 6 ## TODO: find a better way than to hard code these limits
                elif X=='Cu':
                    x1, x2, y1, y2 = 1.2, 2.1, 4.8, 5.9
                else:
                    x1, x2, y1, y2 = 1.0, 1.9, 4.9, 6
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                plt.yticks(visible=False)
                plt.xticks(visible=False)
                ## add zoom effect:
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            #####
            ax.legend(loc='lower right',fontsize=fntsize)
            plt.savefig(f"kinetic_{X}_T.pdf",format='pdf',bbox_inches='tight')
            plt.close()
            
        if plotrho:  
            ## compare rho
            print("rho")
            fig, ax = plt.subplots(1, 1, figsize=(4.5,3.5))
            ax.set_ylabel(r'$\mathrm{log}_{10}(\dot\epsilon_\mathrm{p}/\mathrm{s}^{-1})$',fontsize=fntsize)
            ax.set_xlabel(r'$\mathrm{log}_{10}(\hat\sigma/\sigma_c)$',fontsize=fntsize)
            new_xmax=int(min(x_max,strainrate(X, sigx_max, 1e12, 1e12, 2*rho_0[X], 300)+1,xmax(X,1e12,2*rho_0[X],T=300,rhoi=1e12)+1)) # stop plot at next higher integer past log10 of the maximum strain rate (if any) at given rhom
            # ax.axis((sigx_min,sigx_max,x_min,new_xmax))
            ax.axis((-1,sigx_max,-1,new_xmax))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_title(f"{X}",fontsize=fntsize)
            for rh in [rho_0[X],1.5*rho_0[X],2*rho_0[X]]:
                straindot = np.zeros(siga_res) ## reset
                for x,sigax in enumerate(siga):
                    straindot[x] = strainrate(X, sigax, 1e12, 1e12, rh, 300)
                ax.plot(siga,straindot,label=r'$\rho='+f"{rh/rho_0[X]:.1f}"+r'\rho_0$')
                # print("#pts with low-stress approx: {}, with high-stress approx: {}, A={:.4f}".format(len(sigratio[sigratio<0.35]),len(sigratio[sigratio>100]),Afct(X, 1e12, rh, 300)))
            ax.legend(loc='lower right',fontsize=fntsize)
            plt.savefig(f"kinetic_{X}_rho0.pdf",format='pdf',bbox_inches='tight')
            plt.close()
