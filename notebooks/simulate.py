import traits.api as tr
import numpy as np
import statsmodels.api as sm
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec
from nipype.interfaces.traits_extension import isdefined
import scipy.stats as stats
import prettyplotlib as ppl
import seaborn as sns
from IPython.display import Image
import matplotlib.pyplot as plt
import prettyplotlib as ppl


class simInput(BaseInterfaceInputSpec):
    """
    We define a set of parameters to generate subjects at a given site. 
    Each site has healthy controls (hc), patients with MS (ms) and
    multisite controls (mhc). We also must define an EDSS distribution
    at each site, as well as individual and scanner variabilities. Finally,
    we define relationships between our dependent variable (brain volume BV) and
    age, edss etc.
    """
    # Age ranges
    age_hc_range = tr.Tuple(tr.Float(20.0),tr.Float(50.0),usedefault=True)
    age_ms_range = tr.Tuple(tr.Float(20.0),tr.Float(50.0),usedefault=True)
    age_mhc_range = tr.Tuple(tr.Float(20.0),tr.Float(50.0),usedefault=True)
    
    #Number of subjects
    n_hc = tr.Int(50,usedefault=True)
    n_ms = tr.Int(250,usedefault=True)
    n_mhc = tr.Int(10,usedefault=True)
    n_mhc_scans = tr.Int(2,usedefault=True)
    
    # Distribution parameters
    d_dist = tr.Float(0.88,usedefault=True)
    sig_b = tr.Float(57,usedefault=True)
    sig_s = tr.Float(16,usedefault=True)
    
    # Coefficients
    alpha_a = tr.Float(-3,usedefault=True)
    alpha_d = tr.Float(-16,usedefault=True)
    b0 = tr.Float(1644,usedefault=True)
    alpha = tr.Float(0.9,usedefault=True)
    gamma = tr.Float(20,usedefault=True)
    
    #HC Bias parameters
    alpha_hc = tr.Float(1,usedefault=True)
    gamma_hc = tr.Float(0,usedefault=True)
    
    #EDSS Distribution parameters
    lo_mean = tr.Float(2.0,usedefault=True)
    hi_mean = tr.Float(6.0,usedefault=True)
    lo_sig = tr.Float(1.1,usedefault=True)
    hi_sig = tr.Float(0.8,usedefault=True)
    
    #Name
    sid = tr.String("Site1")

class simOutput(TraitedSpec):

    df = tr.Any()
    scan = tr.Tuple(desc="args to Function that takes real BV and returns scanned BV")
    
    def _repr_html_(self):
        return self.df.describe()._repr_html_()
    
    
def generate_real_BV(ages,alpha_a=-3,b0=1644,edss=None,alpha_d=-16,sig_b = 57):
    if edss == None:
        edss = np.zeros(len(ages))
    BV_real = alpha_a*ages + alpha_d*edss + np.random.normal(b0,sig_b,len(ages))
    return BV_real

def get_scan_function(alpha,gamma,sig_s):
    def scanner_func(x): 
        return alpha * x + np.random.normal(gamma,sig_s,len(x))
    return scanner_func

class sim(BaseInterface):
    input_spec = simInput
    output_spec = simOutput
    
    def _run_interface(self,runtime):
        return runtime
    
    def _sim_data(self):
        o = {}#self._outputs().get()
        o['age_hc'] = np.linspace(*self.inputs.age_hc_range,
                                          num = self.inputs.n_hc)
        o['age_ms'] = np.linspace(*self.inputs.age_ms_range,
                                  num = self.inputs.n_ms)
        o['edss'] = (np.random.binomial(1,self.inputs.d_dist,self.inputs.n_ms)*\
                    np.random.normal(self.inputs.lo_mean,self.inputs.lo_sig,self.inputs.n_ms) +\
                    np.random.binomial(1,1-self.inputs.d_dist,self.inputs.n_ms)*\
                    np.random.normal(self.inputs.hi_mean,self.inputs.hi_sig,
                                     self.inputs.n_ms))
        o['edss'][o['edss']<0] = 0 #EDSS Cannot be below 0
        o['edss'][o['edss']>10] = 10 #EDSS Cannot be >10
        
        Pr_lo = stats.norm.pdf((o["edss"] - 2)/1.1)
        Pr_hi = stats.norm.pdf((o["edss"] - 6)/0.8)
        o["llr"] = np.log(Pr_lo/Pr_hi) > 0


        BV_hc_real = generate_real_BV(o['age_hc'],
                                      self.inputs.alpha_a,
                                      self.inputs.b0,
                                      None,
                                      self.inputs.alpha_d,
                                      self.inputs.sig_b)
                    
        BV_hc_real = self.inputs.alpha_hc * BV_hc_real + self.inputs.gamma_hc
        o['BV_hc'] = self.inputs.alpha * BV_hc_real + \
                       self.inputs.gamma + np.random.normal(0,self.inputs.sig_s,
                                                            self.inputs.n_hc)

        BV_ms_real = generate_real_BV(o['age_ms'],
                                      self.inputs.alpha_a,
                                      self.inputs.b0,
                                      o['edss'],
                                      self.inputs.alpha_d,
                                      self.inputs.sig_b)

        o['BV_ms'] = self.inputs.alpha * BV_ms_real + self.inputs.gamma + \
                       np.random.normal(0,self.inputs.sig_s, self.inputs.n_ms)
            
        return o

    def get_df(self,o):
        #o = self.get()
        n_ms = self.inputs.n_ms
        n_hc = self.inputs.n_hc
        p = {}
        p['subject'] = ["%s_%03d"%(self.inputs.sid,d) for d in range(n_ms+n_hc)]
        p['subject_type'] = np.hstack((["MS"]*n_ms,["HC"]*n_hc))
        p['age'] = np.hstack((o['age_ms'],o['age_hc']))
        p['edss'] = np.hstack((o['edss'],np.ones(n_hc)*np.nan))
        p['edss_group'] = np.hstack((o['llr'],np.ones(n_hc)*np.nan))
        p['BV'] = np.hstack((o['BV_ms'],o['BV_hc']))
        df = pd.DataFrame(data=p)
        self.df = df
        return df
    
    def _list_outputs(self):
        o = self._outputs().get()
        p = self._sim_data()
        
        o['df'] = self.get_df(p)
        o['scan'] = (self.inputs.alpha,
                     self.inputs.gamma,
                     self.inputs.sig_s)

        return o

def kdeplot(Y,df,col=None,groupby=None):
    if not isinstance(Y,str):
        raise Exception("Use seaborn kdeplot")
    if col is not None:
        vals = np.unique(df[col])
        n = len(vals)
        fig,ax = plt.subplots(1,n,figsize=(5*n,5))
        for i ,val in enumerate(vals):
            kdegroup(Y,df[df[col]==val],label=val,groupby=groupby,ax=ax[i])
    else:
        fig, ax= plt.subplots(1)
        kdegroup(Y,df,label="",groupby=groupby,ax=ax)
    return ax

def kdegroup(Y,df,label,groupby,ax):
    if groupby is not None:
        vals = np.unique(df[groupby])
        n = len(vals)
        for i,group in enumerate(vals):
            sns.kdeplot(np.asarray(df[Y][df[groupby]==group].tolist()),
                        ax=ax,label=group,color=ppl.set2[i],linewidth=3)
        ax.set_title(label)
        ax.set_xlabel(Y)
    else:
        sns.kdeplot(np.asarray(df[Y].tolist()))
    

class combineInput(BaseInterfaceInputSpec):
    sites = tr.List(tr.Any())
    site_names = tr.List(tr.String("Site 1"))
    scanners = tr.List(tr.Function())
    
    #Multisite Controls get scanned at each site
    age_msc = tr.List(tr.Float(),[24,24,24,30,32,35,37,40,45,50],usedefault=True)
    n_scans = tr.Int(2,usedefault=True)
    sig_b = tr.Float(57,usedefault=True)
    alpha_a = tr.Float(-3,usedefault=True)
    alpha_d = tr.Float(-16,usedefault=True)
    b0 = tr.Float(1644,usedefault=True)
    

class combineOutput(TraitedSpec):
    df = tr.Any()
    
    def _repr_html_(self):
        return self.df.describe()._repr_html_()
    
    def plot_BV_dist(self):
        df = self.df
        kdeplot("BV",self.df,groupby="protocol",col="subject_type");
        
    def plot_edss_dist(self):
        df = self.df
        ax = kdeplot("edss",df[df["subject_type"]=="MS"],groupby="protocol")
        ax.set_title("EDSS Distributions");

class combine(BaseInterface):
    input_spec = combineInput
    output_spec = combineOutput
    
    def _check_inputs(self):
        num_sites = len(self.inputs.sites)
        num_names = len(self.inputs.site_names)
        if not num_names == num_sites:
            self.inputs.site_names = ["Site %d"%(i+1) for i in range(num_sites)]
    
    def _run_multisite_controls(self):
        #age_msc = np.array([24,24,24,30,32,35,3,40,45,50])
        BV_msc_real = generate_real_BV(np.asarray(self.inputs.age_msc),
                                       alpha_a=self.inputs.alpha_a,
                                       b0=self.inputs.b0,
                                       edss=None,
                                       alpha_d=self.inputs.alpha_d,
                                       sig_b =self.inputs.sig_b)
        
        for i, site in enumerate(self.inputs.sites):
            scan_args = self.inputs.scanners[i]
            scan = get_scan_function(*scan_args)
            p = {}
            n_msc = len(self.inputs.age_msc)
            p['age'] = np.asarray(self.inputs.age_msc)
            p['edss'] = np.ones(n_msc)*np.nan
            p['edss_group'] = np.ones(n_msc)*np.nan
            p['protocol'] = [self.inputs.site_names[i]]*n_msc
            p['subject_type']=["HC"]*n_msc
            p['subject'] = ["eurotrip_%03d"%(d) for d in range(n_msc)]
            p['BV'] = scan(BV_msc_real)
            tmp_df = pd.DataFrame(data=p)
            if not i:
                msc_df = tmp_df
            else:
                msc_df = msc_df.append(tmp_df,ignore_index=True)
        return msc_df
    
    def _run_interface(self,runtime):
        
        self._check_inputs()
        
        for i, site in enumerate(self.inputs.sites):
            n = len(site)
            if not i:
                df = site
                protocol = [self.inputs.site_names[i]]*n
            else:
                df = df.append(site,ignore_index=True)
                protocol = np.hstack((protocol,
                                        [self.inputs.site_names[i]]*n))
        msc_df = self._run_multisite_controls()
        df["protocol"] = protocol
        self._df = df.append(msc_df,ignore_index=True)
        
        return runtime
    
    def _list_outputs(self):
        o = self._outputs().get()
        o["df"] = self._df
        return o

def write_csv(df,name="data.csv"):
    import os
    fname = os.path.abspath(name)
    df.to_csv(fname)
    return fname
