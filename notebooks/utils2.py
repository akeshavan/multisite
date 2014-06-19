#!/usr/bin/python
# These utils model the interaction terms (i.e. age*dis for patients)

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
        ppl.plot(ax[i],x,y,linewidth=0,marker="o")
        coef = np.polyfit(x,y,1)
        ppl.plot(ax[i],x,[coef[0]*f +coef[1] for f in x],linewidth=2)
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
    o['s1_ms'] = run_model([o['age_ms_1'],o['dis_1'],o['age_ms_1']*o['dis_1']],
                            o['BV_ms_1'],sm.RLM,write,yname="BV_1",xnames=["const","age","EDSS","age x EDSS"])
    if write: print "\nSite 2\n"
    o['s2_ms'] = run_model([o['age_ms_2'],o['dis_2'],o['age_ms_2']*o['dis_2']],
                            o['BV_ms_2'],sm.RLM,write,yname="BV_2",xnames=["const","age","EDSS","age x EDSS"])

    if plot:
        plot_coef_split([o['s1_ms'].params,o['s2_ms'].params],
                [o['s1_ms'].bse,o['s2_ms'].bse],
                ["const","age","disease"],
                ["Site 1","Site 2"])


def run_combined_models(o,plot=False,write=False,int_0=True):
    o['combine_v1_ms'] = run_model([o['age_all_ms'],o['dis_all'],o['dis_age_all']],o['BV_all_ms'],sm.RLM,write)
    o['combine_v2_ms'] = run_model([o['age_all_ms'],o['dis_all'],o['dis_age_all'],o['protocol_ms']],o['BV_all_ms'],sm.RLM,write)
    o['combine_z_calib'] = run_model([o["dis_all"],o["dis_age_all"]],o["BV_all_z"],sm.RLM,write,int_0)
    #o['combine_v3'] = run_model([o['age_all_ms'],o['dis_all']],o['BV_all_calib'],sm.RLM, write)
    if plot:
        plot_coef_split([o['combine_v1_ms'].params, o['combine_v2_ms'].params[:3]],#, o['combine_v3'].params],
          [o['combine_v1_ms'].bse, o['combine_v2_ms'].bse[:3]],#, o['combine_v3'].bse],
          ["const","age","disease"],
          ["Combined","Protocol"],#, "Calibrated"],
          cmap = ppl.set2[2:])




def sim_data(n_hc,n_ms,age_hc_range,age_ms_range,
             alpha_a,alpha_d,b0,sig_b,d_dist,
             sig_s,alpha,gamma,alpha_i):

    o = {}

    o['age_hc'] = linspace(*age_hc_range,num = n_hc)
    o['age_ms_1'] = linspace(*age_ms_range[0],num = n_ms[0])
    o['age_ms_2'] = linspace(*age_ms_range[1],num = n_ms[1])

    o['dis_1'] = np.random.binomial(1,d_dist[0],n_ms[0])*np.random.normal(2,1.1,n_ms[0]) +\
                    np.random.binomial(1,1-d_dist[0],n_ms[0])*np.random.normal(6,0.8,n_ms[0])
    o['dis_1'][o['dis_1']<0] = 0 #EDSS Cannot be below 0
    o['dis_1'][o['dis_1']>10] = 10 #EDSS Cannot be >10

    o['BV_hc_1_real'] = alpha_a*o['age_hc'] + np.random.normal(b0,sig_b[0],n_hc)
    o['BV_hc_1'] = alpha[0] * o['BV_hc_1_real'] + gamma[0] + np.random.normal(0,sig_s[0],n_hc)

    o['BV_ms_1_real'] = alpha_a*o['age_ms_1'] + np.random.normal(b0,sig_b[0],n_ms[0]) + alpha_d[0] * o['dis_1'] \
                        + alpha_i[0] * o['age_ms_1'] * o["dis_1"]
    o['BV_ms_1'] = alpha[0] * o['BV_ms_1_real'] + gamma[0] + np.random.normal(0,sig_s[0],n_ms[0])

    o['dis_2'] = np.random.binomial(1,d_dist[1],n_ms[1])*np.random.normal(2,1.1,n_ms[1]) +\
                 np.random.binomial(1,1-d_dist[1],n_ms[1])*np.random.normal(6,0.8,n_ms[1])
    o['dis_2'][o['dis_2']<0] = 0 #EDSS Cannot be below 0
    o['dis_2'][o['dis_2']>10] = 10 #EDSS Cannot be >10

    o['BV_hc_2_real'] = alpha_a*o['age_hc'] + np.random.normal(b0,sig_b[1],n_hc)
    o['BV_hc_2'] = alpha[1]* o["BV_hc_2_real"] +  gamma[1] + np.random.normal(0,sig_s[1],n_hc)

    o['BV_ms_2_real'] = alpha_a*o['age_ms_2'] + np.random.normal(b0,sig_b[1],n_ms[1]) + alpha_d[1] * o['dis_2']\
                        + alpha_i[1] * o['age_ms_2'] * o["dis_2"]
    o['BV_ms_2'] = alpha[1] * o['BV_ms_2_real'] + gamma[1] + np.random.normal(0,sig_s[1],n_ms[1])

    o['BV_all_ms'] = np.hstack((o['BV_ms_1'],o['BV_ms_2']))
    
    o['dis_all'] = np.hstack((o['dis_1'],o['dis_2']))
    o['age_all_ms'] = np.hstack((o['age_ms_1'],o['age_ms_2']))
    o['dis_age_all'] = o['dis_all']*o['age_all_ms']

    o['age_all_hc'] = np.hstack((o['age_hc'],o['age_hc']))
    o['BV_all_hc'] = np.hstack((o['BV_hc_1'],o['BV_hc_2']))
    
    o['BV_all_all'] = np.hstack((o['BV_all_ms'],o['BV_all_hc']))
    o['age_all_all'] = np.hstack((o['age_all_ms'],o['age_all_hc']))
    o['dis_all_all'] = np.hstack((o['dis_all'],np.zeros(n_hc*2)))
    o['dis_age_all_all'] = o['dis_all_all']*o['age_all_all']
    
    
    o['protocol_ms'] = np.hstack((np.ones(n_ms[0]),np.zeros(n_ms[1])))
    o['protocol_all'] = np.hstack((o['protocol_ms'],np.ones(n_hc),np.zeros(n_hc)))
    o['subject_type'] = np.hstack((np.ones(sum(n_ms)),np.zeros(n_hc*2)))

    return o


def grid(o,n_range,parameter,n_boot=10):
    # These are for the z calibration model
    z_coefs = np.zeros((3,len(n_range),n_boot))
    z_errs = np.zeros((3,len(n_range),n_boot))
    z_z = np.zeros((3,len(n_range),n_boot))
    z_conf = np.zeros((3,2,len(n_range),n_boot))
    #These are for the protocol combined model
    p_coefs = np.zeros((5,len(n_range),n_boot))
    p_errs = np.zeros((5,len(n_range),n_boot))
    p_z = np.zeros((5,len(n_range),n_boot))
    p_conf = np.zeros((5,2,len(n_range),n_boot))
    #Combined calibration
    cc_coefs = np.zeros((6,len(n_range),n_boot))
    cc_errs = np.zeros((6,len(n_range),n_boot))
    cc_z = np.zeros((6,len(n_range),n_boot))
    cc_conf = np.zeros((6,2,len(n_range),n_boot))
    
    for i,n in enumerate(n_range):
        for j in range(n_boot):
            
            # Set parameter
            o[parameter] = n
            
            # Run full model
            sim6 = sim_data(**o)
            run_calib(sim6)
            
            #run_hc_model(sim6)
            run_separate_models(sim6)
            run_combined_models(sim6)
            all_calib = run_model([sim6["age_all_all"],sim6["dis_all_all"],sim6["dis_age_all_all"],
                       sim6["subject_type"],sim6["protocol_all"]],
                      sim6["BV_all_all"],sm.RLM,write=False)
            
            # Grab parameters from z calibrated data
            coef = sim6["combine_z_calib"].params
            err = sim6["combine_z_calib"].bse
            z_coefs[:,i,j] = coef
            z_errs[:,i,j] = err
            z_conf[:,:,i,j] = sim6["combine_z_calib"].conf_int()
            z_z[:,i,j] = sim6["combine_z_calib"].tvalues
            
            # Grab parameters from protocol calibrated data
            coef_real = sim6["combine_v2_ms"].params
            err_real = sim6["combine_v2_ms"].bse
            p_coefs[:,i,j] = coef_real
            p_errs[:,i,j] = err_real
            p_conf[:,:,i,j] = sim6["combine_v2_ms"].conf_int()
            p_z[:,i,j] = sim6["combine_v2_ms"].tvalues
            
            coef = all_calib.params
            err = all_calib.bse
            cc_coefs[:,i,j] = coef
            cc_errs[:,i,j] = err
            cc_conf[:,:,i,j] = all_calib.conf_int()
            cc_z[:,i,j] = all_calib.tvalues
        
    return z_coefs, z_errs, z_conf, z_z, p_coefs, p_errs, p_conf, p_z,cc_coefs, cc_errs,cc_conf,cc_z



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
        sns.boxplot(coefs[i,:,:].T,color="pastel",ax=a)
        a.set_xticklabels(d_range)
        a.set_xlabel(d_range_name)
        a.set_title(coef_names[i]);
    
    suptitle(title,size=14)
    return ax
