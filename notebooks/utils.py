import prettyplotlib as ppl
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
np.random.seed(9876789)
from scipy.stats import zscore
from matplotlib.pylab import *
import seaborn as sns
from copy import deepcopy
import pandas as pd
import rpy2.robjects as ro
eq = ro.packages.importr("equivalence")
import scipy.stats as stats
from IPython.display import HTML
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import matplotlib as mpl
mpl.rc("text",usetex=True)
from statsmodels.distributions.empirical_distribution import ECDF
import sklearn.linear_model as lm

def plot_dist(vars,names,title=''):
    k = len(names)
    fig,ax = subplots(1,k, figsize=(4*k,4))
    for i in range(k):
        m = np.mean(vars[i])
        s = np.std(vars[i])
        at = get_label("$\mu$ = %2.2f\n$\sigma$ = %2.2f"%(m,s))
        if not k ==1:
            ppl.hist(ax[i],vars[i])
            ax[i].set_xlabel(names[i])
            ax[i].add_artist(at)
        else:
            ppl.hist(ax,vars[i])
            ax.set_xlabel(names[i])
            ax.add_artist(at)
    suptitle(title,size=14)

def get_label(txt,loc=2):
    at = AnchoredText(txt,
                      prop=dict(size=12), frameon=True,
                      loc=loc,
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    return at

def plot_corr(xi,y,xnames,yname,title="",ax=None):
    if ax == None:
        fig,ax = subplots(1,len(xi),figsize=(5*len(xi),4))
    
    if len(xi)==1: 
        ax=[ax]
    for i,x in enumerate(xi):
        #ppl.plot(ax[i],x,y,linewidth=0,marker="o")
        sns.regplot(x,y,ax=ax[i])
        coef = np.polyfit(x,y,1)
        #ppl.plot(ax[i],x,[coef[0]*f +coef[1] for f in x],linewidth=2)
        ax[i].set_xlabel(xnames[i])
        ax[i].set_ylabel(yname)
        txt = "%s = %2.2e * %s + %2.2f"%(yname, coef[0], xnames[i],coef[1])
        at = get_label(txt)
        ax[i].add_artist(at)
    suptitle(title,size=14)

def run_model(xi,y, model=sm.OLS,write=True,add_intercept=True,yname=None,xnames=None,title=None):
    k = len(xi)
    X = xi[0]
    for i in range(1,k):
        X = np.column_stack((X,xi[i]))
    if add_intercept:
        X = sm.add_constant(X)
    m = model(y,X)
    if model == sm.Logit:
        res = m.fit(disp=False)
    else:
        res = m.fit()
    if write:
        print res.summary(yname=yname,xname=xnames,title=title)
    return res

def plot_coef(coef_list,err_list,coef_name_list,coef_site_list,title="",cmap=ppl.set2):
    fig, ax = subplots(1,1)
    N = len(coef_list[0])
    M = len(coef_list)
    width = 1./(N+1)
    ax.set_xlim(0,M+1)
    for i in range(M):
        X = np.arange(N)+i*width
        ppl.bar(ax,X,coef_list[i],yerr = err_list[i],label=coef_site_list[i],width=width,color=cmap[i])
    
    ax.set_xticks(np.arange(N)+width*N/2.)
    ax.set_xticklabels(coef_name_list)
    ppl.legend(ax)

def plot_coef_split(coef_list,err_list,coef_name_list,coef_site_list,title="",cmap=ppl.set2):
    N = len(coef_list) # num sites
    M = len(coef_list[0]) # num coefs
    
    fig, ax = subplots(1,M,figsize=(3*M,4)) #One plot per coef
    width = 1./(N)
    for j in range(M): #for each coef
        a = ax[j]
        a.set_xlim(.2,1.2)
        for i in range(N): #for each site
            X = [i*width] #there is one coef
            
            ppl.bar(a,X,[coef_list[i][j]],
                    yerr = [err_list[i][j]],
                    label=coef_site_list[i],
                    width=width,color=cmap[i],annotate=True)
    
        a.set_xticks([])
        a.set_xticklabels([])
        a.set_title(coef_name_list[j])
    
    ppl.legend(a)


def run_calib(o,plot=False,write=False):
    o['hc_1_calib'] = run_model([o['age_hc']],o['BV_hc_1'],sm.OLS,write)
    o['hc_2_calib'] = run_model([o['age_hc']],o['BV_hc_2'],sm.OLS,write)

    # Calibration is predicted @ site1 vs predicted @ site2
    o['calib'] = run_model([o["hc_1_calib"].fittedvalues],o["hc_2_calib"].fittedvalues,sm.OLS,write)
    #             run_model([o['BV_hc_1']],o['BV_hc_2'],sm.OLS,write)

    o['resid_1'] = o['hc_1_calib'].fittedvalues - o['BV_hc_1']
    o['resid_2'] = o['hc_2_calib'].fittedvalues - o['BV_hc_2']


    o['BV_hc_1_calib_trans'] = o['calib'].params[1] * o ['BV_hc_1'] + o['calib'].params[0]
    o['hc_12_calib'] = run_model([o['BV_hc_1_calib_trans']],o['BV_hc_2'],sm.OLS,write)

    o['BV_ms_1_calib'] = o['calib'].params[1] * o['BV_ms_1'] + o['calib'].params[0]

    o['BV_ms_1_z'] = np.divide(o['BV_ms_1'] - o['hc_1_calib'].predict(sm.add_constant(o['age_ms_1'])),o['hc_1_calib'].params[1])
    o['BV_ms_2_z'] = np.divide(o['BV_ms_2'] - o['hc_2_calib'].predict(sm.add_constant(o['age_ms_2'])),o['hc_2_calib'].params[1])


    o['BV_all_calib'] = np.hstack((o['BV_ms_1_calib'],o['BV_ms_2']))
    o['BV_all_z'] = np.hstack((o['BV_ms_1_z'],o['BV_ms_2_z']))
    

    
    if plot:
        fig, ax = subplots(1,2,figsize=(10,4))
        #plot_corr([o['BV_hc_1']],o['BV_hc_2'],
        #          ["$\hat{BV_1}$"],
        #          "$\hat{BV_2}$",
        #          "",ax[0])

        plot_corr([o["age_hc"]],o["BV_hc_1"],["age"],"$BV_{hc,1}$","",ax[0])
        plot_corr([o["age_hc"]],o["BV_hc_2"],["age"],"$BV_{hc,2}$","",ax[1])
        ax[0].set_title("Site 1 Controls")
        ax[1].set_title("Site 2 Controls")
        return ax
        #ymin = min(min(o["BV_hc_1"]),min(o["BV_hc_2"]))
        #ymax = max(max(o["BV_hc_1"]),max(o["BV_hc_2"]))
        #ax[0].set_ylim([ymin,ymax])
        #ax[1].set_ylim([ymin,ymax])
        #plot_corr([o["hc_1_calib"].fittedvalues],
        #          o["hc_2_calib"].fittedvalues,
        #          ["$\dot{BV_1}$"],
        #          "$\dot{BV_2}$",
        #          "",ax[1])

        #plot_dist([o['BV_ms_1'],o['BV_ms_2'],o['BV_ms_1_calib']],["BV1","BV2","BV_1_calib"])
        #plot_dist([o['resid_1'],o['resid_2']],["r1","r2"],title="Residuals")
        #plot_dist([o['BV_ms_1_z'],o['BV_ms_2_z']],["z1","z2"],title="Z Scored BV")


def plot_calib(sim):
    plot_dist([o['BV_ms_1_z'],o['BV_ms_2_z']],["z1","z2"],title="Z Scored BV")


def run_separate_models(o,plot=False,write=False):

    if write: print "Site 1\n"
    o['s1_ms'] = run_model([o['age_ms_1'],o['dis_1']],o['BV_ms_1'],sm.RLM,write,yname="BV_1",xnames=["const","age","EDSS"])
    if write: print "\nSite 2\n"
    o['s2_ms'] = run_model([o['age_ms_2'],o['dis_2']],o['BV_ms_2'],sm.RLM,write,yname="BV_2",xnames=["const","age","EDSS"])

    if plot:
        plot_coef_split([o['s1_ms'].params,o['s2_ms'].params],
                [o['s1_ms'].bse,o['s2_ms'].bse],
                ["const","age","disease"],
                ["Site 1","Site 2"])


def run_combined_models(o,plot=False,write=False,int_0=True,model=sm.RLM):
    o['combine_v1_ms'] = run_model([o['age_all_ms'],o['dis_all']],o['BV_all_ms'],model,write)
    o['combine_v2_ms'] = run_model([o['age_all_ms'],o['dis_all'],o['protocol_ms']],o['BV_all_ms'],model,write)
    o['combine_z_calib'] = run_model([o["dis_all"]],o["BV_all_z"],model,write,int_0)
    o['combine_p_calib'] = run_model([o["age_all_ms"],o["dis_all"]],o["pBV_ms_all"],model,write,int_0)
    #o['combine_v3'] = run_model([o['age_all_ms'],o['dis_all']],o['BV_all_calib'],sm.RLM, write)
    """if plot:
        plot_coef_split([o['combine_v1_ms'].params, o['combine_v2_ms'].params[:3]],#, o['combine_v3'].params],
          [o['combine_v1_ms'].bse, o['combine_v2_ms'].bse[:3]],#, o['combine_v3'].bse],
          ["const","age","disease"],
          ["Combined","Protocol"],#, "Calibrated"],
          cmap = ppl.set2[2:])"""

def run_combined_log_models(o,plot=False,write=False,int_0=True,model=sm.Logit):
    o['combine_v1_ms'] = run_model([o['age_all_ms'],o['BV_all_ms']],o['llr_all'],model,write)
    o['combine_v2_ms'] = run_model([o['age_all_ms'],o['BV_all_ms'],o['protocol_ms']],o['llr_all'],model,write)
    o['combine_z_calib'] = run_model([o["BV_all_z"]],o["llr_all"],model,write,int_0)
    o['combine_p_calib'] = run_model([o["age_all_ms"],o["pBV_ms_all"]],o["llr_all"],model,write,int_0)
 

def to_percentile(sim,ref_key,key):
    foo = ECDF(sim[ref_key],side="right")
    return foo(sim[key])



def sim_data(n_hc,n_ms,age_hc_range,age_ms_range,
             alpha_a,alpha_d,b0,sig_b,d_dist,
             sig_s,alpha,gamma, alpha_hc=None,
             gamma_hc=None,lo_mean=[2,2],lo_sig=[1.1,1.1],
             hi_mean=[6,6],hi_sig=[0.8,0.8]):

    o = {}

    o['age_hc'] = linspace(*age_hc_range,num = n_hc)
    o['age_ms_1'] = linspace(*age_ms_range[0],num = n_ms[0])
    o['age_ms_2'] = linspace(*age_ms_range[1],num = n_ms[1])

    o['dis_1'] = np.random.binomial(1,d_dist[0],n_ms[0])*np.random.normal(lo_mean[0],lo_sig[0],n_ms[0]) +\
                    np.random.binomial(1,1-d_dist[0],n_ms[0])*np.random.normal(hi_mean[0],hi_sig[0],n_ms[0])
    o['dis_1'][o['dis_1']<0] = 0 #EDSS Cannot be below 0
    o['dis_1'][o['dis_1']>10] = 10 #EDSS Cannot be >10
    
    o['BV_hc_1_real'] = alpha_a*o['age_hc'] + np.random.normal(b0,sig_b[0],n_hc)
    if alpha_hc is not None and gamma_hc is not None:
        o['BV_hc_1_real'] = alpha_hc[0]*o['BV_hc_1_real'] + gamma_hc[0]
    
    o['BV_hc_1'] = alpha[0] * o['BV_hc_1_real'] + gamma[0] + np.random.normal(0,sig_s[0],n_hc)

    o['BV_ms_1_real'] = alpha_a*o['age_ms_1'] + np.random.normal(b0,sig_b[0],n_ms[0]) + alpha_d[0] * o['dis_1']
    o['BV_ms_1'] = alpha[0] * o['BV_ms_1_real'] + gamma[0] + np.random.normal(0,sig_s[0],n_ms[0])

    o['dis_2'] = np.random.binomial(1,d_dist[1],n_ms[1])*np.random.normal(lo_mean[1],lo_sig[1],n_ms[1]) +\
                 np.random.binomial(1,1-d_dist[1],n_ms[1])*np.random.normal(hi_mean[1],hi_sig[1],n_ms[1])
    o['dis_2'][o['dis_2']<0] = 0 #EDSS Cannot be below 0
    o['dis_2'][o['dis_2']>10] = 10 #EDSS Cannot be >10

    o['BV_hc_2_real'] = alpha_a*o['age_hc'] + np.random.normal(b0,sig_b[1],n_hc)
    if alpha_hc is not None and gamma_hc is not None:
        o['BV_hc_2_real'] = alpha_hc[1]*o['BV_hc_2_real'] + gamma_hc[1]
        
    o['BV_hc_2'] = alpha[1]* o["BV_hc_2_real"] +  gamma[1] + np.random.normal(0,sig_s[1],n_hc)

    o["Pr_1_lo"] = stats.norm.pdf((o["dis_1"] - 2)/1.1)
    o["Pr_1_hi"] = stats.norm.pdf((o["dis_1"] - 6)/0.8)
    o["llr_1"] = log(o["Pr_1_lo"]/o["Pr_1_hi"]) > 0

    o["Pr_2_lo"] = stats.norm.pdf((o["dis_2"] - 2)/1.1)
    o["Pr_2_hi"] = stats.norm.pdf((o["dis_2"] - 6)/0.8)
    o["llr_2"] = log(o["Pr_2_lo"]/o["Pr_2_hi"]) > 0

    o['BV_ms_2_real'] = alpha_a*o['age_ms_2'] + np.random.normal(b0,sig_b[1],n_ms[1]) + alpha_d[1] * o['dis_2']
    o['BV_ms_2'] = alpha[1] * o['BV_ms_2_real'] + gamma[1] + np.random.normal(0,sig_s[1],n_ms[1])

    o['BV_all_ms'] = np.hstack((o['BV_ms_1'],o['BV_ms_2']))

    o['dis_all'] = np.hstack((o['dis_1'],o['dis_2']))
    o['age_all_ms'] = np.hstack((o['age_ms_1'],o['age_ms_2']))

    o['age_all_hc'] = np.hstack((o['age_hc'],o['age_hc']))
    o['dis_all'] = np.hstack((o['dis_1'],o['dis_2']))
    o['BV_all_hc'] = np.hstack((o['BV_hc_1'],o['BV_hc_2']))

    o['BV_all_all'] = np.hstack((o['BV_all_ms'],o['BV_all_hc']))
    o['age_all_all'] = np.hstack((o['age_all_ms'],o['age_all_hc']))
    o['dis_all_all'] = np.hstack((o['dis_all'],np.zeros(n_hc*2)))

    o['protocol_ms'] = np.hstack((np.ones(n_ms[0]),np.zeros(n_ms[1])))
    o['protocol_all'] = np.hstack((o['protocol_ms'],np.ones(n_hc),np.zeros(n_hc)))
    o['subject_type'] = np.hstack((np.ones(sum(n_ms)),np.zeros(n_hc*2)))

    return o

def run_p_calib(o):
    o["pBV_hc_1"] = to_percentile(o,"BV_hc_1","BV_hc_1")
    o["pBV_hc_2"] = to_percentile(o,"BV_hc_2","BV_hc_2")
    o["pBV_ms_1"] = to_percentile(o,"BV_hc_1","BV_ms_1")
    o["pBV_ms_2"] = to_percentile(o,"BV_hc_2","BV_ms_2")
    o["pBV_ms_all"] = np.hstack((o["pBV_ms_1"],o["pBV_ms_2"]))
    o["llr_all"] = np.hstack((o["llr_1"],o["llr_2"]))   
    #o["llr_all_all"] = np.hstack((o["llr_all"], np.zeros(len(o["pBV_hc_1"])+len(o["pBV_hc_2"]))))
    o['pBV_all_hc'] = np.hstack((o['pBV_hc_1'],o['pBV_hc_2']))
    o['pBV_all_ms'] = np.hstack((o['pBV_ms_1'],o['pBV_ms_2']))
    o['pBV_all_all'] = np.hstack((o['pBV_all_ms'],o['pBV_all_hc']))




def grid(o,n_range,parameter,n_boot=10,model=sm.RLM,combine_type=run_combined_models):
    # These are for the z calibration model
    z_coefs = np.zeros((2,len(n_range),n_boot))
    z_errs = np.zeros((2,len(n_range),n_boot))
    z_z = np.zeros((2,len(n_range),n_boot))
    z_conf = np.zeros((2,2,len(n_range),n_boot))
    #These are for the p calibrated model
    p_coefs = np.zeros((3,len(n_range),n_boot))
    p_errs = np.zeros((3,len(n_range),n_boot))
    p_z = np.zeros((3,len(n_range),n_boot))
    p_conf = np.zeros((3,2,len(n_range),n_boot))

    #Combined Protcol calibration (z) or logistic
    if not model==sm.Logit: num = 5
    else: num = 4
    cc_coefs = np.zeros((num,len(n_range),n_boot))
    cc_errs = np.zeros((num,len(n_range),n_boot))
    cc_z = np.zeros((num,len(n_range),n_boot))
    cc_conf = np.zeros((num,2,len(n_range),n_boot))

    # Combined Protocol+Percentile Calibration
    ccp_coefs = np.zeros((5,len(n_range),n_boot))
    ccp_errs = np.zeros((5,len(n_range),n_boot))
    ccp_z = np.zeros((5,len(n_range),n_boot))
    ccp_conf = np.zeros((5,2,len(n_range),n_boot))
 
    for i,n in enumerate(n_range):
        for j in range(n_boot):
            
            # Set parameter
            o[parameter] = n
            
            # Run full model
            sim6 = sim_data(**o)
            run_calib(sim6)
            run_p_calib(sim6)
            
            #run_hc_model(sim6)
            #run_separate_models(sim6)
            combine_type(sim6,model)
            if not model == sm.Logit:
                all_calib = run_model([sim6["age_all_all"],sim6["dis_all_all"],
                       sim6["subject_type"],sim6["protocol_all"]],
                      sim6["BV_all_all"],model,write=False)
            else:
                all_calib = sim6["combine_v2_ms"]
                #all_calib = run_model([sim6["age_all_all"],sim6["BV_all_all"],
                #       sim6["subject_type"],sim6["protocol_all"]],
                #      sim6["llr_all_all"],model,write=False)
      


            # Grab parameters from z calibrated data
            coef = sim6["combine_z_calib"].params
            err = sim6["combine_z_calib"].bse
            z_coefs[:,i,j] = coef
            z_errs[:,i,j] = err
            z_conf[:,:,i,j] = sim6["combine_z_calib"].conf_int()
            z_z[:,i,j] = sim6["combine_z_calib"].tvalues

            # Grab parameters from p calibrated data
            coef = sim6["combine_p_calib"].params
            err = sim6["combine_p_calib"].bse
            p_coefs[:,i,j] = coef
            p_errs[:,i,j] = err
            p_conf[:,:,i,j] = sim6["combine_p_calib"].conf_int()
            p_z[:,i,j] = sim6["combine_p_calib"].tvalues
             
            # All z calib          
            coef = all_calib.params
            err = all_calib.bse
            cc_coefs[:,i,j] = coef
            cc_errs[:,i,j] = err
            cc_conf[:,:,i,j] = all_calib.conf_int()
            cc_z[:,i,j] = all_calib.tvalues

            # All p calib
            """coef = all_p_calib.params
            err = all_p_calib.bse
            ccp_coefs[:,i,j] = coef
            ccp_errs[:,i,j] = err
            ccp_conf[:,:,i,j] = all_p_calib.conf_int()
            ccp_z[:,i,j] = all_p_calib.tvalues"""
         
    return (z_coefs, z_errs, z_conf, z_z), (p_coefs, p_errs, p_conf, p_z),(cc_coefs, cc_errs,cc_conf,cc_z)


def plot_grid(coefs,errs,hc_age_range,name,title="Calibration Coefficients"):
    fig,ax = subplots(1,2,figsize=(10,4))
    
    for j in range(coefs.shape[-1]):
        ax[0].errorbar(hc_age_range,coefs[0,:,j],color=ppl.set1[0],marker='o',yerr=errs[0,:,j]);
        ax[1].errorbar(hc_age_range,coefs[1,:,j],color=ppl.set1[1],marker='o',yerr=errs[1,:,j]);
    
    suptitle(title,size=14)
    ax[0].set_xlabel(name)
    ax[1].set_xlabel(name)

    ax[0].set_ylabel("Intercept")
    ax[1].set_ylabel("Disease")

def plot_grid_box(coefs,d_range,coef_names,title="",d_range_name="",ax=None):
    n_coefs = coefs.shape[0]
    if ax == None:
        fig,ax = subplots(1,n_coefs,figsize=(5*n_coefs,5))
    if n_coefs == 1: ax = [ax]
    for i,a in enumerate(ax):
        sns.boxplot(coefs[i,:,:].T,ax=a,color="Set2")
        a.set_xticklabels(d_range)
        a.set_xlabel(d_range_name)
        a.set_title(coef_names[i]);

    suptitle(title,size=14)
    return ax

def init_df(params):
    columns = ['delta_z']
    for key,item in params.iteritems():
        if isinstance(item,list) or isinstance(item,tuple):
            for idx,thing in enumerate(item) or isinstance(item,tuple):
                if isinstance(thing,list) or isinstance(thing,tuple):
                    for j,r in enumerate(thing):
                        newname = key+"_site_%d"%idx+"_%s"%(["lo","hi"][j])
                        #print newname 
                        columns.append(newname)
                        
                else:
                    newname = key+"_site_%d"%idx
                    #print newname
                    columns.append(newname)
        else:
            newname = key
            #print newname
            columns.append(newname)
    df = pd.DataFrame(columns = columns)
    return df

def set_df(params,df):
    newobject = {}
    for key,item in params.iteritems():
        if isinstance(item,list) or isinstance(item,tuple):
            for idx,thing in enumerate(item):
                if isinstance(thing,list) or isinstance(thing,tuple):
                    for j,r in enumerate(thing):
                        newname = key+"_site_%d"%idx+"_%s"%(["lo","hi"][j])
                        newobject[newname] = r
                        
                else:
                    newname = key+"_site_%d"%idx
                    newobject[newname] = thing
        else:
            newname = key
            newobject[newname] = item
    df = df.append(newobject,ignore_index=True)        
    if df:
        return df
    else:
        raise Exception("Something went wrong!!")

def add_grid(diff_dis,key,a,n_ms_range,df):
    delta_z = np.mean(diff_dis,axis=1)
    for i, n in enumerate(n_ms_range):
        a[key] = n
        a['delta_z'] = delta_z[i]
        df = set_df(a,df)
    return df

params = {'n_hc':50, #number of controls
          'n_ms':[250,100], #number of MS patients at each site
          'age_hc_range':(20,52), # Age range of the controls
          'age_ms_range':[(20,45),(20,75)], # Age range of MS patients at each site
          'd_dist':[.65,.88], # Distribution weighting of EDSS scores for each site
          'alpha_a':-3, # Real coefficient of age
          'alpha_d':[-16,-16], # Real coefficient of EDSS score
          'b0':1644, #Brain Volume Intercept (in ccs)
          'sig_b':[57,57], # Brain variation across subjects (in ccs) for each site
          'sig_s':[16,16], # Variability in brain volume measurement across scanners for each site
          'alpha':[0.7,1.3], # Scaling of each scanner
          'gamma':[20,-20] # Offset of each scanner
          }

def run_sim(age_euro_orig=np.array([24,24,24,38,37,32,28,30,32,32,50]),
         params=params,n_scans=2,keep_subs=False):
    sim = sim_data(**params)    
    n_euro = len(age_euro_orig)
    BV_euro_real_orig = params['alpha_a']*age_euro_orig + \
                        np.random.normal(params['b0'],params['sig_b'][0],n_euro)
    BV_euro_real = BV_euro_real_orig; 
    age_euro=age_euro_orig
            
    BV_euro_1 = params['alpha'][0]*BV_euro_real+\
                np.random.normal(params['gamma'][0],params['sig_s'][0]*1/sqrt(n_scans),n_euro)
    BV_euro_2 = params['alpha'][1]*BV_euro_real+\
                np.random.normal(params['gamma'][1],params['sig_s'][1]*1/sqrt(n_scans),n_euro)
        
    if keep_subs:
        sim['BV_hc_1'] = np.hstack((sim['BV_hc_1'],BV_euro_1))
        sim['BV_hc_2'] = np.hstack((sim['BV_hc_2'],BV_euro_2))
        sim['age_hc'] = np.hstack((sim['age_hc'],age_euro))
         
    run_calib(sim,False)
    #z_site_1 = (sim['BV_hc_1'] - sim['hc_1_calib'].fittedvalues)/\
    #     np.std(sim['BV_hc_1'] - sim['hc_1_calib'].fittedvalues)
    #z_site_2 = (sim['BV_hc_2'] - sim['hc_2_calib'].fittedvalues)/\
    #     np.std(sim['BV_hc_2'] - sim['hc_2_calib'].fittedvalues)
    
    z_site_1 = sim["calib"].predict(sm.add_constant(sim["BV_hc_1"]))
    z_site_2 = sim["BV_hc_2"]
    
    if keep_subs:
        z_euro_1 = z_site_1[-n_euro:]
        z_euro_2 = z_site_2[-n_euro:]
        
    else:
        #euro_1 = (BV_euro_1 - sim['hc_1_calib'].predict(sm.add_constant(age_euro)))
        #euro_2 = (BV_euro_2 - sim['hc_2_calib'].predict(sm.add_constant(age_euro)))
        
        z_euro_1 = sim["calib"].predict(sm.add_constant(BV_euro_1))
        z_euro_2 = BV_euro_2
        
        #z_euro_1 = euro_1/np.std(euro_1)
        #z_euro_2 = euro_2/np.std(euro_2)
    
    zdiff = np.abs(z_euro_1-z_euro_2)/z_euro_2 #Now delta can be in units of percent
    return zdiff

def run_percentile_sim(age_euro_orig=np.array([24,24,24,38,37,32,28,30,32,32,50]),
         params=params,n_scans=2,keep_subs=False):
    sim = sim_data(**params)    
    n_euro = len(age_euro_orig)
    BV_euro_real_orig = params['alpha_a']*age_euro_orig + \
                        np.random.normal(params['b0'],params['sig_b'][0],n_euro)
    BV_euro_real = BV_euro_real_orig; 
    age_euro=age_euro_orig
            
    BV_euro_1 = params['alpha'][0]*BV_euro_real+\
                np.random.normal(params['gamma'][0],params['sig_s'][0]*1/sqrt(n_scans),n_euro)
    BV_euro_2 = params['alpha'][1]*BV_euro_real+\
                np.random.normal(params['gamma'][1],params['sig_s'][1]*1/sqrt(n_scans),n_euro)
        
    if keep_subs:
        sim['BV_hc_1'] = np.hstack((sim['BV_hc_1'],BV_euro_1))
        sim['BV_hc_2'] = np.hstack((sim['BV_hc_2'],BV_euro_2))
        sim['age_hc'] = np.hstack((sim['age_hc'],age_euro))
         
    run_p_calib(sim)
    
    z_site_1 = sim["pBV_hc_1"]
    z_site_2 = sim["pBV_hc_2"]
    
    if keep_subs:
        z_euro_1 = z_site_1[-n_euro:]
        z_euro_2 = z_site_2[-n_euro:]
        
    else:
        sim["BV_euro_1"] = BV_euro_1
        sim["BV_euro_2"] = BV_euro_2
        z_euro_1 = to_percentile(sim,"BV_hc_1","BV_euro_1")
        z_euro_2 = to_percentile(sim,"BV_hc_2","BV_euro_2")
    
    zdiff = np.abs(z_euro_1 - z_euro_2)
    #np.abs(z_euro_1-z_euro_2)/z_euro_2 #Now delta can be in units of percent
    return zdiff



def boot(n_boot,age_euro_orig=np.array([24,24,24,38,37,32,28,30,32,32,50]),
         params=params,n_scans=2,keep_subs=False,sim_type=run_sim):
    
    Diffs = np.zeros((n_boot,len(age_euro_orig)))
    
    for i in range(n_boot):
        zdiff = sim_type(age_euro_orig,params,n_scans,keep_subs)
        Diffs[i,:] = zdiff
    return Diffs

def eq_test(pdiff,delta):
    import rpy2.robjects as ro
    eq = ro.packages.importr("equivalence")
    import numpy as np

    x = ro.FloatVector(pdiff)
    base = ro.packages.importr("base")
    e = delta/std(pdiff)
    res = eq.ptte_data(x,Epsilon=e)
    P = res.rx("Power")[0][0]
    T = res.rx("Tstat")[0][0]
    C = res.rx("cutoff")[0][0]
    return P,T,C


def get_table(a):
    data = [["Parameter","Site 1","Site 2"]]
    for key, val in sorted(a.iteritems()):
        if isinstance(val,list):
            data.append([key, "%s"%str(val[0]), "%s"%str(val[1])])
        else:
            data.append([key,str(val),str(val)])
    txt = """\\begin{tabular}{llll}"""
    txtend = """\\end{tabular}"""
    for i,row in enumerate(data):
        txt+=" & ".join(row).replace("_"," ")
        txt+=("""\\\ \\hline """)
    txt += txtend
    return txt

def anisha():
    return None

def test_all(pdiff,test_E = arange(0.001,0.01,0.0005)):
    import rpy2.robjects as ro
    eq = ro.packages.importr("equivalence")
    import numpy as np

    def eq_test(pdiff,delta):
        x = ro.FloatVector(pdiff)
        base = ro.packages.importr("base")
        e = delta/std(pdiff)
        res = eq.ptte_data(x,Epsilon=e)
        P = res.rx("Power")[0][0]
        T = res.rx("Tstat")[0][0]
        C = res.rx("cutoff")[0][0]
        return P,T,C

    Ps = np.zeros(len(test_E))
    Ts = np.zeros(len(test_E))
    Cs = np.zeros(len(test_E))
    #test_E = arange(0.25,1,0.05)
    for i,e in enumerate(test_E):
        p,t,c = eq_test(pdiff,e)
        Ps[i] = p
        Cs[i] = c
        Ts[i] = t
    return Ps, Ts, Cs

def draw_interp(x,y,ax,**kwargs):
    f2 = InterpolatedUnivariateSpline(x, y)
    ppl.plot(ax,linspace(min(x),max(x),100),
             f2(linspace(min(x),max(x),100)),
             **kwargs)

def run_log_model(xi,y,add_intercept=True):
    import sklearn.linear_model as lm
    a = lm.LogisticRegression()
    k = len(xi)
    X = xi[0]

    for i in range(1,k):
        X = np.column_stack((X,xi[i]))
    if add_intercept:
        X = sm.add_constant(X)

    a.fit(X,y)
    
    #ROC
    roc_info = map(get_ROC,[X]*100,[y]*100)
    roc_info = np.asarray(roc_info)
    fpr = roc_info[:,0]
    tpr = roc_info[:,1]
    auc = roc_info[:,2]

    return np.mean(auc)

def get_ROC(X,y):
    from sklearn.metrics import roc_curve, auc
    from sklearn.utils import shuffle
    import sklearn.linear_model as lm
    X, y = shuffle(X, y)
    half = int(len(y) / 2)
    X_train, X_test = X[:half], X[half:]
    y_train, y_test = y[:half], y[half:]

    # Run classifier
    classifier = lm.LogisticRegression()
    classifier.fit(X_train, y_train)
    probas_ = classifier.predict_proba(X_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,roc_auc


