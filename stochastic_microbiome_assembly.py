#!/usr/bin/env python3
#
# Version 1.0
# This python program consitutes companion code to the paper ``Stochastic
# community assembly of the Drosophila melanogaster microbiome depends on
# context.'' 
#
# This code:
#   + imports raw microbiome abundance data from a combinatorial dissection fly
#   experiment (originally published in Gould et al., PNAS 2018) from the file
#   raw_combinatorial_microbiome_abundance_data.csv
#   + showcases the variable colonization outcomes across fly experiments
#   + constructs and evaluates independent models of colonization
#   + produces Figs. 2-4 of the manuscript
#
# The code is covered under GNU GPLv3.
# Send any questions or comments to Eric W. Jones at jones.eric93@gmail.com
###############################################################################

import numpy as np
import itertools
import functools
import math
import statsmodels.stats.proportion
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import operator
import pickle
import scipy
import scipy.stats as stats
import scipy.cluster.hierarchy as hierarchy
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import statsmodels.stats.weightstats as weightstats



#############################################
### INPUT/OUTPUT AND MANIPULATION OF RAW DATA
#############################################

def import_fly_data(samples_per_exp='all'):
    """Purpose: Imports combinatorial raw fly microbiome abundance data
       Input: samples_per_exp ... if 'all' (default), sample_dict contains all
                data; if an integer, sample_dict contains samples_per_exp
                bootstrapped samples for each combination
       Returns: sample_dict ... dictionary full of abundance data
                  e.g. sample_dict['11010'] = [[A0, B0, C0, D0, E0],
                                               [A1, B1, C1, D1, E1], ...]"""

    filename = 'raw_combinatorial_fly_microbiome_abundance_data.csv'
    with open(filename,'r') as f:
        all_data = [line.strip().split(",") for line in f][1:]

    sample_dict = {}
    for sample in all_data:
        # list_comb ... fed bacterial combination, e.g. [1, 1, 0, 1, 0]
        list_comb = [int(val) for val in sample[1:6]]
        # str_comb ... fed bacterial combination, e.g. '11010'
        str_comb = ''.join([str(val) for val in list_comb])
        # comb_abundances ... bacterial abundance vector for this sample
        comb_abundances = [float(val) for val in sample[7:12]]

        # populate sample_dict
        try:
            sample_dict[str_comb].append(comb_abundances)
        except KeyError:
            sample_dict[str_comb] = []
            sample_dict[str_comb].append(comb_abundances)

    # in the DEFAULT case, return the experimentally measured abundances
    if samples_per_exp == 'all':
        return sample_dict

    # OTHERWISE, construct sample_dict with bootstrapped (synthetic) data
    bootstrapped_sample_dict = {}
    for comb in sample_dict:
        num_samples = len(sample_dict[comb])
        bootstrapped_indices = np.random.randint(0, num_samples,
                                                 samples_per_exp)
        bootstrapped_sample_dict[comb] = []
        for idx in bootstrapped_indices:
            bootstrapped_sample_dict[comb].append(sample_dict[comb][idx])

    return bootstrapped_sample_dict

def get_prob_outcomes(sample_dict=import_fly_data()):
    """Purpose: construct dict of colonization probabilities (via
                presence/absence data)
       Input:   sample_dict ... from import_fly_data()
       Returns: prob_dict ... dictionary full of presence/absence data for
                  all possible subsets of a bacterial combination
                  e.g. prob_dict['10010'] = {(0, 1): p1, (0): p2, (1): p3, 
                                             (): p4}
                  (note p1+p2+p3+p4 must equal 1.0) """

    # colonize_counts will hold 'presence' data
    colonize_counts = {}
    for key in sample_dict:
        diversity = sum([int(x) for x in key])
        # generate all 2^N possible outcomes for a diversity-N experiment
        outcomes = powerset(range(diversity))
        colonize_counts[key] = {outcome: 0 for outcome in outcomes}
        indices = [i for i,x in enumerate(key) if x == '1']

        for j,sample in enumerate(sample_dict[key]):
            # abundances ONLY of those species that were fed
            ordinal_sample = [sample[i] for i in indices]
            # convert abundance -> presence/absence data
            nonzero_elements = [1 if val else 0 for val in ordinal_sample]
            # collect indices of nonzero elements
            nonzero_indices = tuple([i for i,val in enumerate(nonzero_elements) if val])
            colonize_counts[key][nonzero_indices] += 1

    # turn raw count data into proportions
    prob_dict = {}
    for key in colonize_counts:
        prob_dict[key] = {}
        num_samples = len(sample_dict[key])
        for outcome in colonize_counts[key]:
            prob_dict[key][outcome] = colonize_counts[key][outcome]/num_samples

    return prob_dict

#############################################
### HELPER FUNCTIONS
#############################################

def get_bacteria_labels():
    """ Returns a list of bacterial species in the order in which abundance
    data was provided in import_fly_data() """
    return ['LP', 'LB', 'AP', 'AT', 'AO']

def powerset(iterable):
    """ Function from Friedman et al., Nature 2017
    Purpose: return powerset of an iterable list
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def get_max_likelihood_independent_model_probs(sample_dict=import_fly_data(),
                                               fit_data_w_diversity='all', verbose=False):
    """Identify the marginal probabilities of the independent model that
    maximizes the log-likelihood of observing the data.
    Inputs: verbose ... boolean indicating whether to output diagnostic text
            fit_data_w_diversity ... int 2-5 or 'all'; describes which dataset
                                     the model will be fit to
    Returns: max_likelihood_probs ... length 5 vector of marginal probabilities """

    # assemble data
    prob_dict = get_prob_outcomes(sample_dict)

    # compute optimal marginal probabilities via optimization
    # minimize error = maximize log-likelihood
    p_opt = scipy.optimize.shgo(get_error_from_independent_model,
                                bounds=[(0, 1.0),]*5,
                                args=(sample_dict,fit_data_w_diversity),
                                options={'disp': verbose})

    return p_opt.x

def get_error_from_independent_model(ps_guess, sample_dict=import_fly_data(),
                                     fit_data_w_diversity='all'):
    """ Compute the negative log-likehood that an independent model using a particular
    set of marginal probabilities (ps_guess) matches the observed
    presence/absence patterns of bacterial species across every fly experiment.
    Inputs: ps_guess ... candidate length-5 vector of marginal probabilities
            fit_data_w_diversity ... int 2-5 or 'all'; describes which dataset
                                     the model will be fit to
    Returns: negative_log_likelihood ... float """

    prob_dict = get_prob_outcomes(sample_dict)
    num_samples = len(sample_dict['10000'])

    log_likelihood_error = 0

    for key in prob_dict:
        N = sum([int(elem) for elem in key])
        num_subcombs = len(prob_dict[key])

        if fit_data_w_diversity != 'all':
            if N != fit_data_w_diversity:
                continue

        # get independent-model-predicted + observed multinomial distributions
        predicted_distribution, observed_distribution = (
            get_predicted_and_observed_multinomial_distribution(
                key, ps_guess, sample_dict, prob_dict))

        key_log_likelihood = get_log_likelihood(observed_distribution,
                                                predicted_distribution)
        log_likelihood_error += key_log_likelihood

    negative_log_likelihood = -1.0*log_likelihood_error
    return negative_log_likelihood

def get_error_from_interaction_colonization_model(params_guess,
                                                  fit_data_w_diversity,
                                                  sample_dict=import_fly_data()):
    """ Compute the negative log-likehood that an independent model using a particular
    set of marginal probabilities (ps_guess) matches the observed
    presence/absence patterns of bacterial species across every fly experiment.
    Inputs: params_guess ... candidate length-9 vector of params (marginal
                probs + interaction params) to fit
            fit_data_w_diversity ... int 2-5 or 'all'; describes which dataset
                the model will be fit to. if 'all', all data is used to fit the
                five marginal probabilities and interaction
                boosts/penalties. if an integer, the marginal probabilities
                will be from get_marginal_probs_for_each_diversity, and the
                interaction parameters will be fit to data of
                this diversity.
    Returns: negative_log_likelihood ... """

    prob_dict = get_prob_outcomes(sample_dict)
    num_samples = len(sample_dict['10000'])

    if fit_data_w_diversity != 'all':
        marginal_probs, marginal_prob_errs = (
                get_marginal_probs_for_each_diversity(sample_dict=import_fly_data()))
        empirical_probs = np.array(marginal_probs[fit_data_w_diversity])
        params_guess = np.concatenate((empirical_probs, params_guess))

    log_likelihood_error = 0

    for key in prob_dict:
        N = sum([int(elem) for elem in key])
        # only use certain experiments to fit parameters
        if fit_data_w_diversity != 'all':
            if N != fit_data_w_diversity:
                continue

        num_subcombs = len(prob_dict[key])

        # get independent-model-predicted + observed multinomial distributions
        predicted_distribution, observed_distribution = (
            get_interaction_multinomial_distributions(
                key, params_guess, sample_dict, prob_dict))

        key_log_likelihood = get_log_likelihood(observed_distribution,
                                                predicted_distribution)
        log_likelihood_error += key_log_likelihood

    negative_log_likelihood = -1.0*log_likelihood_error
    return negative_log_likelihood

def get_predicted_and_observed_multinomial_distribution(
        key, ps_guess, sample_dict, prob_dict):
    """ Compute the expected and observed multinomial distributions of
    num_samples trials, in which the probability of each outcome are determined
    by an independent context-independent model with marginal probabilities ps_guess.
    Inputs: key ... e.g. '11010', set of fed bacteria for which the multinomial
                    distribution is computed
            ps_guess ... length-5 vector of marginal probabilities for the
                    independent model of interest
    Returns: model_distribution ... length-2^N vector (N = diversity of key)
                    describing presence/absence configuration of fed species;
                    sum(model_distribution) = num_samples
             observed_distribution ... length-2^N vector of observed
                    presence/absence distribution"""

    N = sum([int(elem) for elem in key])
    num_samples = len(sample_dict['10000'])
    num_subcombs = len(prob_dict[key])

    model_distribution = []
    observed_distribution = []

    for subkey in prob_dict[key]:
        # subcombo_prob is the probability of observing that particular subkey.
        # thus each subcombo_prob is the probability of a particular category
        # in a multinomial distribution.
        subcombo_prob = 1
        indices = [i for i,elem in enumerate([int(val) for val in key]) if elem]

        for i in range(N):
            if i in subkey:
                subcombo_prob = subcombo_prob*ps_guess[indices[i]]
            else:
                subcombo_prob = subcombo_prob*(1-ps_guess[indices[i]])

        model_distribution.append(subcombo_prob)
        observed_distribution.append(prob_dict[key][subkey]*num_samples)

    return model_distribution, observed_distribution

def get_interaction_multinomial_distributions(
        key, params_guess, sample_dict, prob_dict):
    """ Compute the expected and observed multinomial distributions of
    num_samples trials, in which the probability of each outcome are determined
    by a interaction model with marginal probabilities params_guess[:5] and
    interactions params_guess[5:].
    Inputs: key ... e.g. '11010', set of fed bacteria for which the multinomial
                    distribution is computed
            params_guess ... candidate length-9 vector of params (marginal
                    probs + interaction params) to fit
    Returns: model_distribution ... length-2^N vector (N = diversity of key)
                    describing presence/absence configuration of fed species;
                    sum(model_distribution) = num_samples
             observed_distribution ... length-2^N vector of observed
                    presence/absence configuration """

    ps_guess = params_guess[:5]
    interaction = params_guess[5:]

    ps_interaction = get_interaction_probs(key, ps_guess, interaction)

    N = sum([int(elem) for elem in key])
    num_samples = len(sample_dict['10000'])
    num_subcombs = len(prob_dict[key])

    model_distribution = []
    observed_distribution = []

    for subkey in prob_dict[key]:
        subcombo_prob = 1
        indices = [i for i,elem in enumerate([int(val) for val in key]) if elem]

        for i in range(N):
            if i in subkey:
                subcombo_prob = subcombo_prob*ps_interaction[indices[i]]
            else:
                subcombo_prob = subcombo_prob*(1-ps_interaction[indices[i]])

        model_distribution.append(subcombo_prob)
        observed_distribution.append(prob_dict[key][subkey]*num_samples)

    return model_distribution, observed_distribution


def get_log_likelihood(obsv_dist, model_dist):
    """ Compute the log-likelihood that num_samples trials of a multinomial
    distribution with category probabilities given by model_dist would be
    distributed like obsv_dist.
    Inputs: obsv_dist ... list of the number of times each outcome occurred
                          (len(obsv_dist) = num_samples)
            model_dist ... probability of each outcome occurring
    Returns: log_likelihood ... """

    # clean up inputs
    model_dist = [val/sum(model_dist) for val in model_dist]
    obsv_dist = [int(val) for val in obsv_dist]
    num_samples = sum(obsv_dist)

    # compute probability of this multinomial distribution
    p = (math.factorial(num_samples)
         /functools.reduce(operator.mul, [math.factorial(val) for val in obsv_dist], 1)
         * np.product([x**y for x,y in zip(model_dist, obsv_dist)]))

    if p > 0:
        return np.log(p)
    else:
        return -np.inf

def compute_multinomial_test(obsv_dist, model_dist, xnomial, n_trials):
    """ Compute an exact or Monte Carlo multinomial test to determine how
    likely the observed distribution (or a less likely distribution) would be
    under a multinomial distribution null model whose outcome probabilities
    are given by model_dist.
    Inputs: obsv_dist ... list of the number of times each outcome occurred
                          (len(obsv_dist) = num_samples)
            model_dist ... probability of each outcome occurring
            n_trials ... number of trials to use in Monte Carlo multinomial test
    Returns: p_val ... probability of observing the observed distribution under
                       the null model"""

    r_obsv_dist = robjects.FloatVector(obsv_dist)
    r_model_dist = robjects.FloatVector(model_dist)

    # if possible, use the exact multinomial test (using the R package xnomial)
    if len(obsv_dist) <= 4:
        ans = xnomial.xmulti(r_obsv_dist, r_model_dist, detail=0)
        pval = ans.rx2('pProb')[0]
    # else, use the Monte Carlo (approximate) multinomial test 
    else:
        ans = xnomial.xmonte(r_obsv_dist, r_model_dist, statName='Prob',
                             detail=0, ntrials=n_trials)
        pval = ans.rx2('pProb')[0]

    return pval

def get_interaction_probs(key, ps_guess, interaction):
    """ Return marginal probabilities adjusted to take into account the context
    of which bacteria were fed alongside them, according to the interaction
    model.
    Inputs: key ... bacterial combination
            ps_guess ... length-5 vector of marginal probabilities for basic
                         independent model
            interaction ... length-4 vector of parameters describing inter- and
                         intragenus interactions
    Returns: ps_interaction ... length-5 vector of adjusted marginal
                               probabilities (taking interactions
                               between bacteria into account) """

    indices = [i for i,elem in enumerate([int(val) for val in key]) if elem]
    N = sum([int(elem) for elem in key])

    ps_interaction = np.copy(ps_guess)

    # segregate bacterial indices into Acetobacter genus (indices 0 and 1) and
    # Lactobacillus species (indices 2, 3, and 4)
    in_group_a = []
    in_group_b = []
    for index in indices:
        if index <= 1:
            in_group_a.append(index)
        else:
            in_group_b.append(index)

    AA = len(in_group_a)
    BB = len(in_group_b)
    for index in indices:
        if (index in in_group_a) and len(in_group_a) > 1:
            ps_interaction[index] = ps_interaction[index]**(interaction[0]**1)
        if (index in in_group_a) and len(in_group_b) > 0:
            ps_interaction[index] = ps_interaction[index]**(interaction[1]**1)
        if (index in in_group_b) and len(in_group_b) > 1:
            ps_interaction[index] = ps_interaction[index]**(interaction[2]**1)
        if (index in in_group_b) and len(in_group_a) > 0:
            ps_interaction[index] = ps_interaction[index]**(interaction[3]**1)

    return ps_interaction

#############################################
### PLOTTING HELPER FUNCTIONS
#############################################

def add_legend(ax, add_x=0, add_y=0):
    """Decorates ax with a legend
        Input: ax
               add_x ... shift to x
               add_y ... shift to y
        Returns: ax """
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from matplotlib.container import ErrorbarContainer

    legends = []
    colors = ['#ff6300ff', '#ffd400ff', '#b02effff', '#009cf0ff', '#00cc0cff']
    labels = ['LP', 'LB', 'AP', 'AT', 'AO', '95\% CI']

    legend_elements = [Line2D([0], [0], color=colors[i], lw=0, marker='.',
                              mec='k', mew=1.1, ms=25,
                              label=labels[i]) for i in range(5)]
    # from https://stackoverflow.com/questions/56074771/add-errorbar-to-marker-in-line2d-element-for-legend
    line = Line2D([],[], ls='none', color='k', lw=2)
    barline = LineCollection(np.empty((2,2,2)), color='k', lw=1.5)
    err = ErrorbarContainer((line, [line], [barline]), has_xerr=False,
                            has_yerr=True, label=labels[-1])

    legend_elements.append(err)

    leg = ax.legend(handles=legend_elements, fontsize=10, loc=(0.35+add_x,
                                                               1.02+add_y),
                    ncol=6, handletextpad=0.3, borderpad=0.5, framealpha=1, frameon=True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_edgecolor('k')

    return ax

def plot_pies_on_points(ax, keys, x_vals, y_vals, if_xlabel=False, colors=None,
                        off_color='black', my_circ_size=None, my_lw=None,
                        fudge_factor=1.0):
    """Purpose: decorates plot with colorful pies that indicate bacterial
                outcomes
       Inputs: ax ... axes object on which plot exists
               keys ... list of [key, subkey] pairs indicating order in which
                        to plot pies
               x_vals ... list of x-values (associated with each key) on which
                          to plot pies
               y_vals ... list of y-values (associated with each key) on which
                          to plot pies
               if_xlabel ... whether or not to plot pies on x-axis to specify
                             bacterial combinations
               colors ... order of the 5 colors to use
               off_color ... color to use for combinations in which not all fed
                             species colonize
               my_circ_size ... size to overwrite size of circles
               my_lw ... size to overwrite linewidth
               fudge_factor ... amount to alter aspect_ratio
       Returns: ax ... decorated axes object"""

    # from https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes
    figW, figH = ax.get_figure().get_size_inches()
    _, _, w, h = ax.get_position().bounds
    disp_ratio = (figH * h) / (figW * w)
    data_ratio = operator.sub(*ax.get_ylim()) / operator.sub(*ax.get_xlim())
    aspect_ratio = fudge_factor * data_ratio / disp_ratio

    if (type(colors) == list and any(colors) == None) or isinstance(colors, type(None)):
        colors = ['#ff6300ff', '#ffd400ff', '#b02effff', '#009cf0ff', '#00cc0cff']

    for (key, subkey),x,y in zip(keys,x_vals, y_vals):
        labels = ['LP', 'LB', 'AP', 'AT', 'AO']

        div = sum([int(x) for x in key])
        subdiv = len(subkey)

        # add new pies to plot that will act as x-labels
        if if_xlabel:
            if div == subdiv and y > 0:
                keys.append([key, subkey])
                x_vals.append(x)
                y_vals.append(-0.05)

        indices = [i for i,x in enumerate(key) if x == '1']
        subindices = [indices[i] for i in subkey]

        for i in range(div):
            if indices[i] in subindices:
                color = colors[indices[i]]
            else:
                color = off_color

            if indices == subindices:
                if y >= 0:
                    # for datapoint pies
                    circ_size = .596
                    lw = 1.1
                    zorder = 10
                else:
                    # for xlabel pies
                    circ_size = .43
                    lw = 0.7
                    zorder = 10
            else:
                circ_size = .25
                lw = .6
                zorder = 10+y

            if my_circ_size: circ_size = my_circ_size
            if my_lw: lw = my_lw

            if div == 1:
                m2 = mpatches.Wedge([0, 0], circ_size, 0, 360,
                        color=color, zorder=zorder, ec='black',
                        lw=lw, clip_on=False,
                        transform=(
                            transforms.Affine2D(np.array([[1, 0, 0], [0, aspect_ratio, 0], [0, 0,
                                1]])).translate(x, y) + ax.transData
                            ))
            else:
                m2 = mpatches.Wedge([0, 0], circ_size, 90+(i*360/div),
                        90+((i+1)*360/div), zorder=zorder, color=color,
                        ec='black', lw=lw ,clip_on=False,
                        transform=(
                            transforms.Affine2D(np.array([[1, 0, 0], [0, aspect_ratio, 0], [0, 0,
                                1]])).translate(x, y) + ax.transData
                            ))
            ax.add_patch(m2)
    return ax

def reorder_vals(new_ordering, z, species_labels):
    """Purpose: Permute z according to the new_ordering.
       Inputs: new_ordering ... length-5 list permutation of range(5)
               z ... np.array to permute
               species_labels ... list to permute
       Returns: z ... permuted np.array
                species_labels ... permuted list """
    new_indexing = [species_labels.index(val) for val in new_ordering]
    z = np.array([z[i] for i in new_indexing])
    z = z.T
    z = np.array([z[i] for i in new_indexing])
    z = z.T

    species_labels = np.array([species_labels[i] for i in new_indexing])

    return z, species_labels

def plot_heatmap(z, ax, xlabels, ylabels, cmap='PRGn', cbar_label='',
                 vbound=None, ylabel='model', fontsize=10,
                 xaxis_top=False, rotatey=False, norm=None, dashed=False,
                 cbararrow=False):
    """ Purpose: Plot z as a heatmap on ax.
        Inputs: z ... np.array
                ax ... ax to plot on
                xlabels ... list of strings to serve as xlabels
                ylabels ... list of strings to serve as ylabels
                cmap ... string (colormap) to use in the ax.pcolor command
                cbar_label ... string to set the colormap label
                vbound ... 2-element list [vmin, vmax] for colormap bounds
                ylabel ... string that is the 'super' ylabel
                fontsize ... float
                xaxis_top ... boolean, if true xlabels are on top
                rotatey ... boolean, if true rotate xticklabels
                norm ... if 'log', use log-transformed colormap. otherwise
                         linear colormap is used. 
                dashed ... boolean: if true, insert a line separating the top
                           four colormap rows from the rows below
                cbararrow ... boolean. if true use the 'extend' functionality
                              of fig.colorbar
        Returns: ax ... ax containing fully plotted colormap
    """

    # determine whether xticks are on top or bottom
    if xaxis_top:
        ax.xaxis.tick_top()

    # deal with NaN values in z, determine colormap bounds
    masked_z = np.ma.masked_invalid(z)
    max_val = max(abs(np.nanmax(z)), abs(np.nanmin(z)))
    if vbound:
        vmin = vbound[0]; vmax=vbound[1]
    else:
        vmin = -max_val; vmax = max_val

    # make log-space or linear-space colormap
    if norm == 'log':
        im = ax.pcolor(masked_z, cmap=cmap,
                       norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = ax.pcolor(masked_z, vmin=vmin, vmax=vmax, cmap=cmap)

    # separate context-independent from context-specific models, if needed
    if dashed:
        ax.plot([0, len(xlabels)], [4, 4], ls='-',
                     color='k', lw=1)

    # modify axes
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.patch.set(color='k')

    # set xticks and yticks
    if rotatey: rotation = 0
    else: rotation = 90
    ax.set_xticks([val + 0.5 for val in range(len(xlabels))])
    ax.set_xticklabels(xlabels, rotation=rotation, fontsize=fontsize)
    ax.set_yticks([val + 0.5 for val in range(len(ylabels))])
    ax.set_yticklabels(ylabels, fontsize=fontsize)

    ax.set_xlabel('bacterial combination $k$', labelpad=15, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    # set-up colorbar location on axes and plot colorbar
    fig = plt.gcf()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if cbararrow:
        cbar = fig.colorbar(im, cax=cax, fraction=0.02, pad=0.04, shrink=0.5,
                            extend='both')
    else:
        cbar = fig.colorbar(im, cax=cax, fraction=0.02, pad=0.04, shrink=0.5)
    cbar.set_label(cbar_label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    if norm == 'log':
        cbar.set_ticks([1e-3, 1e-2, 1e-1, 1e0])
    else:
        cbar.set_ticks([np.round(vmin, 2), np.round((vmin+vmax)/2, 2), np.round(vmax, 2)])

    plt.subplots_adjust(top=0.75)

    return ax

#############################################
### GENERATE RESULTS
#############################################

def get_all_colonization_probs(sample_dict=import_fly_data()):
    """Purpose: generate data for Fig. 2 of the paper, which shows the
                  colonization probability for every outcome of every bacterial
                  combination
       Inputs:  sample_dict ... as generated by import_fly_data(), either with
                  'all' data or with some bootstrapped resampling
       Returns: x_vals ... list of 31 x-values
                y_vals ... list of 31 probabilities that all fed species colonize
                errs ... list of 31 tuples, errors in the prob that all fed
                         species colonize
                full_x_vals ... all (jittered) x-values, including sumcombs, to plot 
                full_y_vals ... all colonization probs for all suboutcomes of
                                  all bacterial combinations
                full_y_errs ... all errors (binomial proportion, Jeffreys
                                  method) associated with colonization
                                  probabilities
                full_ordering ... list of all [val, subval] pairs (for use with
                                  plot_pies_on_points function) """

    prob_dict = get_prob_outcomes(sample_dict)
    labels = get_bacteria_labels()

    x_vals = []
    y_vals = []
    errs = []
    full_x_vals = []
    full_y_vals = []
    full_y_errs = []
    full_ordering = []

    for i,val in enumerate(prob_dict):
        if val == '00000':
            continue

        diversity = sum([int(x) for x in val])
        max_index = tuple(range(diversity))

        # get x and y values for the probability that all fed species colonize
        x_vals.append(i + .5)
        y_val = prob_dict[val][max_index]
        y_vals.append(y_val)

        # get x and y values for every subcombination of every bacterial combination 
        num_subvals = len(prob_dict[val])
        for j,subval in enumerate(prob_dict[val]):
            # x_vals chosen to get nice spacing
            if len(subval) == diversity:
                full_x_vals.append(i + .5)
            else:
                full_x_vals.append(i + (j+1)/(num_subvals+1))
            full_y_vals.append(prob_dict[val][subval])
            full_ordering.append([val, subval])

            num_success = prob_dict[val][subval]*len(sample_dict[val])
            num_total = len(sample_dict[val])

            conf_int = statsmodels.stats.proportion.proportion_confint(
                            num_success, num_total,
                            alpha=0.05, method='jeffreys')

            full_y_errs.append([prob_dict[val][subval] - conf_int[0],
                                conf_int[1] - prob_dict[val][subval]])
            if len(subval) == diversity:
                errs.append([prob_dict[val][subval]- conf_int[0], conf_int[1] - prob_dict[val][subval]])

    full_y_errs = np.array(full_y_errs).T
    errs = np.array(errs).T

    return x_vals, y_vals, errs, full_x_vals, full_y_vals, full_y_errs, full_ordering

def get_marginal_probs_for_each_diversity(sample_dict=import_fly_data()):
    """Purpose: generate data for Fig. 3c of the paper, which shows the
                  marginal probability of colonization for each species for each
                  diversity
       Inputs:  sample_dict ... as generated by import_fly_data(), either with
                  'all' data or with some bootstrapped resampling
       Returns: marginal_probs ... dict, call as marginal_probs[diversity][species]
                marginal_prob_errs ... dict, marginal_probs_errs[div][species] 
                                             = [lower_bound, upper_bound] """

    prob_dict = get_prob_outcomes(sample_dict)
    labels = get_bacteria_labels()

    # initialize dicts
    marginal_probs = {diversity: [0 for i in range(5)] for diversity in range(1,6)}
    marginal_prob_errs = {diversity: [[0, 0] for i in range(5)] for diversity in range(1,6)}
    for diversity in range(1,6):
        for species in range(5):
            num_fed = 0
            num_colonized = 0
            for key in sample_dict:
                # only consider combinations that have desired diversity
                key_diversity = sum([int(elem) for elem in key])
                if key_diversity != diversity:
                    continue

                if key[species] == '1':
                    for sample in sample_dict[key]:
                        num_fed += 1
                        if sample[species] > 0:
                            num_colonized += 1

            marginal_probs[diversity][species] = num_colonized/num_fed
            conf_int = statsmodels.stats.proportion.proportion_confint(
                            num_colonized, num_fed,
                            alpha=0.05, method='jeffreys')
            marginal_prob_errs[diversity][species] = (
                            [num_colonized/num_fed - conf_int[0],
                            conf_int[1] - num_colonized/num_fed])

    return marginal_probs, marginal_prob_errs

def get_independent_colonization_model_probs(sample_dict=import_fly_data(),
                                             fit_data_w_diversity='all',
                                             verbose=False, cap_prob=False):
    """ Compute the marginal colonization probabilities for each species for the
    uniform, single-species, two-species, and maximum likehood models
    Inputs: verbose ... boolean indicating whether to output diagnostic text
            fit_data_w_diversity ... int 2-5 or 'all'; describes which dataset
                                     the model will be fit to
            cap_prob ... boolean indicating whether to artificially diminish
                         colonization probabilities of 100% (useful for
                         hypothesis testing)
    Returns: model_labels ... length-4 list of strings containing the names of
                              the independent models 
             independent_probs... 4x5 matrix containing the colonization
                                  probabilities for the four models """
    # independent_probs in the order: uniform, single-species, two-species, and
    # maximum likehood models
    independent_probs = []

    marginal_probs, marginal_prob_errs = (
                get_marginal_probs_for_each_diversity(sample_dict))

    single_species_probs = marginal_probs[1]
    two_species_probs = marginal_probs[2]
    uniform_probs = [np.average(single_species_probs) for i in range(5)]
    max_likelihood_probs = get_max_likelihood_independent_model_probs(
                                sample_dict, fit_data_w_diversity)

    model_labels = ['uniform', 'single-species', 'two-species',
                    'max-likelihood']
    independent_probs.append(uniform_probs)
    independent_probs.append(single_species_probs)
    independent_probs.append(two_species_probs)
    independent_probs.append(max_likelihood_probs)

    if cap_prob:
        max_prob = 0.9999
        for i,probs in enumerate(independent_probs):
            for j,prob in enumerate(probs):
                if prob == 1:
                    independent_probs[i][j] = max_prob

    if verbose:
        for model, probs in zip(model_labels, independent_probs):
            print(model)
            print(' ', probs)
            print('  negative log-likelihood:', get_error_from_independent_model(probs, sample_dict, 'all'))

    return model_labels, np.array(independent_probs)

def get_interaction_params(sample_dict=import_fly_data(),
                           fit_data_w_diversity='all', verbose=False,
                           cap_prob=False, load_data=True):
    """ Compute the base marginal colonization probabilities for each species for the
    interaction model, as well as the context-specific interaction parameters.
    Inputs: fit_data_w_diversity ... int 2-5 or 'all'; describes which dataset
                                     the model will be fit to
            verbose ... boolean indicating whether to output diagnostic text
            cap_prob ... boolean indicating whether to artificially diminish
                         colonization probabilities of 100% (useful for
                         hypothesis testing)
            load_data ... boolean whether to load data
    Returns: base_probs ... length-5 vector of the base marginal probabilities of
                            the interaction model 
             interaction ... length-4 vector containing intra- and intergenus
                           interaction parameters"""

    prob_dict = get_prob_outcomes(sample_dict)

    if load_data:
        try:
            with open('vars/interaction_model_probs_interaction_{}.pi'
                      .format(fit_data_w_diversity) , 'rb') as f:
                base_probs, interaction = pickle.load(f)
            return base_probs, interaction
        except FileNotFoundError:
            print('rerunning fit of optimal parameters for the interaction model')

    if fit_data_w_diversity == 'all':
        # fit all base colonization probs + interaction params at once
        p_opt = scipy.optimize.minimize(get_error_from_interaction_colonization_model,
                                x0=[.9, .9, .9, .9, .9, 1.0, 1.0, 1.0, 1.0],
                                args=(fit_data_w_diversity, sample_dict),
                                bounds=[(0, 1.0),]*5+[(0.1, 5)]*4,
                                method='trust-constr')
        base_probs = p_opt.x[:5]
        interaction = p_opt.x[5:]
    else:
        # only fit interaction params, use base probs from empirical colonization odds
        p_opt = scipy.optimize.minimize(get_error_from_interaction_colonization_model,
                                x0=[1.0, 1.0, 1.0, 1.0],
                                args=(fit_data_w_diversity, sample_dict),
                                method='nelder-mead')
        marginal_probs, marginal_prob_errs = (
                get_marginal_probs_for_each_diversity(sample_dict=import_fly_data()))

        base_probs = np.array(marginal_probs[fit_data_w_diversity])
        interaction = p_opt.x

    with open('vars/interaction_model_probs_interaction_{}.pi'
              .format(fit_data_w_diversity) , 'wb') as f:
        pickle.dump((base_probs, interaction), f)

    if verbose:
        print('interaction {}'.format(fit_data_w_diversity))
        print(' ', base_probs)
        print('  interaction:', interaction)
        print('              alpha_LL, alpha_LA, alpha_AA, alpha_AL')
        print('  negative log-likelihood:',
              get_error_from_interaction_colonization_model(
                  np.concatenate((base_probs, interaction)), 'all', sample_dict))
    return base_probs, interaction

def compute_statistics_of_independent_models(sample_dict=import_fly_data(),
                                             n_trials=5e5, load_data=True,
                                             verbose=False):
    """ For each indepedent model (whose marginal probabilities are generated
    with get_independent_colonization_model_probs), compute exact or Monte
    Carlo multinomial tests for each fed bacterial combination to evaluate
    the probability that the experimentally observed configuration of outcomes
    (or less likely configurations) under a null hypothesis of the multinomial
    distribution (whose outcome probabilities are determined by the marginal
    probabilities of the independent model). 
    Inputs: n_trials ... integer specifying how many trials to use in the Monte
                        Carlo multinomial test
            load_data ... boolean whether to load data
            verbose ... boolean indicating whether to output diagnostic text
    Returns: keys ... length-31 list of combinations (trivial '00000' removed)
             independent_model_pvals ... 4x31 np.array consisting the p-value
                                         for fed bacterial combination for each
                                         of the four independent models
             independent_model_log_likelihoods ... 4x31 np.array consisting the
                                         log-likelihoods for each combination
                                         of fed bacteria for each of the four
                                         independent models"""
    if load_data:
        try:
            with open('vars/independent_model_p_vals_log_likelihoods.pi', 'rb') as f:
                keys, independent_model_pvals, independent_model_log_likelihoods = (
                    pickle.load(f))
            return keys, independent_model_pvals, independent_model_log_likelihoods
        except FileNotFoundError:
            print('rerunning multinomial tests for independent models')

    # import xnomial from R, using importr
    xnomial = importr('XNomial')

    prob_dict = get_prob_outcomes(sample_dict)
    model_labels, independent_probs = (
            get_independent_colonization_model_probs(cap_prob=True))

    keys = list(prob_dict.keys())
    keys.remove('00000')

    independent_model_pvals = np.zeros((len(model_labels), len(keys)))
    independent_model_log_likelihoods = np.zeros((len(model_labels), len(keys)))
    for i,model in enumerate(model_labels):
        if verbose: print('  running simulations for {} model'.format(model))
        for j,key in enumerate(keys):
            model_probs = independent_probs[i]
            predicted_distribution, observed_distribution = (
                get_predicted_and_observed_multinomial_distribution(
                    key, model_probs, sample_dict, prob_dict))

            key_pval = compute_multinomial_test(observed_distribution,
                                                predicted_distribution,
                                                xnomial, n_trials)
            key_log_likelihood = get_log_likelihood(observed_distribution,
                                                    predicted_distribution)

            independent_model_pvals[i][j] = key_pval
            independent_model_log_likelihoods[i][j] = key_log_likelihood

    with open('vars/independent_model_p_vals_log_likelihoods.pi', 'wb') as f:
        pickle.dump((keys, independent_model_pvals, independent_model_log_likelihoods), f)

    return keys, independent_model_pvals, independent_model_log_likelihoods

def compute_statistics_of_interaction_models(
            use_data = ['all', 2], sample_dict=import_fly_data(), n_trials=5e5,
            load_data=True, verbose=False):
    """ For each interaction model (whose marginal probabilities and
    interaction params are generated with get_interaction_params), compute
    exact or Monte Carlo multinomial tests for each fed bacterial combination. 
    Inputs: use_data ... list containing 'all' or int 2-5, to be used as
                         'fit_data_w_diversity' param in other functions
            n_trials ... integer specifying how many trials to use in the Monte
                         Carlo multinomial test
            load_data ... boolean whether to load data
            verbose ... boolean indicating whether to output diagnostic text
    Returns: keys ... length-31 list of keys (trivial '00000' removed)
             interaction_model_pvals ... 4x31 np.array consisting the p-value
                                         for fed bacterial combination for each
                                         of the four independent models
             interaction_model_log_likelihoods ... 4x31 np.array consisting of the
                                         log-likelihoods for fed bacterial
                                         combination for each of the four
                                         independent models"""
    if load_data:
        try:
            with open('vars/interaction_model_p_vals_log_likelihoods.pi', 'rb') as f:
                keys, interaction_model_pvals, interaction_model_log_likelihoods = (
                    pickle.load(f))
            return keys, interaction_model_pvals, interaction_model_log_likelihoods
        except FileNotFoundError:
            print('rerunning multinomial tests for interaction models')

    # import xnomial from R
    xnomial = importr('XNomial')

    prob_dict = get_prob_outcomes(sample_dict)
    keys = list(prob_dict.keys())
    keys.remove('00000')

    interaction_model_pvals = np.zeros((len(use_data), len(keys)))
    interaction_model_log_likelihoods = np.zeros((len(use_data), len(keys)))
    for i,fit_data_w_diversity in enumerate(use_data):
        if verbose: print('  running simulations for interaction {}'
                          .format(fit_data_w_diversity))
        base_probs, interaction = get_interaction_params(
                fit_data_w_diversity=fit_data_w_diversity, verbose=False,
                load_data=False)
        params_guess = np.concatenate((base_probs, interaction))

        for j,key in enumerate(keys):
            predicted_distribution, observed_distribution = (
                get_interaction_multinomial_distributions(
                    key, params_guess, sample_dict, prob_dict))

            key_pval = compute_multinomial_test(observed_distribution,
                                                predicted_distribution,
                                                xnomial, n_trials)
            key_log_likelihood = get_log_likelihood(observed_distribution,
                                                    predicted_distribution)

            interaction_model_pvals[i][j] = key_pval
            interaction_model_log_likelihoods[i][j] = key_log_likelihood

    with open('vars/interaction_model_p_vals_log_likelihoods.pi', 'wb') as f:
        pickle.dump((keys, interaction_model_pvals, interaction_model_log_likelihoods), f)

    return keys, interaction_model_pvals, interaction_model_log_likelihoods

def evaluate_model_performances():
    """ Print out the properties of each independent colonization model,
    as well as the ability of each model to reproduce empirical data (as
    measured by multinomial tests, log likelihoods, and Bayesian Information
    Criterions.
    Returns: void """

    model_labels, independent_probs = (
        get_independent_colonization_model_probs(verbose=False, cap_prob=True,
                                                 fit_data_w_diversity='all'))

    keys, pvals, log_likelihoods = (
        compute_statistics_of_independent_models(verbose=False,
                                                 load_data=True))

    base_probs_all, interaction_all = get_interaction_params(
                             fit_data_w_diversity='all', verbose=False,
                             load_data=True)

    base_probs_2, interaction_2 = get_interaction_params(
                             fit_data_w_diversity=2, verbose=False,
                             load_data=True)

    keys_cf, pvals_cf, log_likelihoods_cf = (
        compute_statistics_of_interaction_models(
                use_data=['all', 2], verbose=False, load_data=True))

    all_base_probs = np.vstack((independent_probs, base_probs_all,
                                base_probs_2))
    all_pvals = np.vstack((pvals, pvals_cf))
    all_log_likelihoods = np.vstack((log_likelihoods, log_likelihoods_cf))
    num_params = [1, 5, 5, 5, 9, 9]
    model_names = ['uniform', 'single-species', 'two-species',
                   'max likelihood', 'interaction (all)', 'interaction (div-2)']

    for i,name in enumerate(model_names):
        print(name)
        print('  species colonization probs:', all_base_probs[i])
        if name == 'interaction (all)':
            print('  alpha_LL, alpha_LA, alpha_AA, alpha_AL:', interaction_all)
        if name == 'interaction (div-2)':
            print('  alpha_LL, alpha_LA, alpha_AA, alpha_AL:', interaction_2)
        print('  num reproduced (p>0.05):',
              sum([val > 0.05 for val in all_pvals[i]]), '/31')
        print('  total log likelihood:', sum(all_log_likelihoods[i]))
        print('  LL by diversity:', sum(all_log_likelihoods[i][:5]),
              sum(all_log_likelihoods[i][5:15]), sum(all_log_likelihoods[i][15:25]),
              sum(all_log_likelihoods[i][25:30]), all_log_likelihoods[i][30])
        print('  BIC:', num_params[i]*np.log(31) - 2*sum(all_log_likelihoods[i]))
        print()

    return


#############################################
### PLOT RESULTS
#############################################

def plot_all_colonization_probs(sample_dict=import_fly_data(),
                                fig_filename='colonization_probabilities.pdf'):
    """Purpose: generate Fig. 2 of the paper, which shows the colonization
                  probability for every outcome of every bacterial combination
       Inputs:  fig_filename ... filename for the saved figure
       Returns: ax ... axes object with probability of colonization plot """

    # assemble data
    prob_dict = get_prob_outcomes(sample_dict)
    x_vals, y_vals, errs, full_x_vals, full_y_vals, full_y_errs, full_ordering = (
            get_all_colonization_probs(sample_dict))

    # initialize environment
    fig, axs = plt.subplots(ncols=5, sharey=True, gridspec_kw={'width_ratios':
                                                             [5,10,10,5,1]})
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08)
    fig.text(0.5, 0.02, 'fed bacterial species', ha='center')
    fig.text(-0.005, 0.5, r'$P$(colonization outcome)', va='center', rotation='vertical')

    # identify subplot limits
    breaks = [0, 5, 15, 25, 30, 31]

    for i,ax in enumerate(axs):
        for j,val in enumerate(prob_dict):
            ax.axvline(x=j+1, linewidth=.2, color='k')

        start = breaks[i]
        end = breaks[i+1]

        # take a slice of the values to plot, and only pass these subset of
        # values to the subplot
        subset_full_xs = []
        subset_full_ys = []
        subset_full_errs = []
        subset_full_ordering = []
        subset_xs = []
        subset_ys = []
        subset_errs = []

        for x,y,err,ordering in zip(full_x_vals, full_y_vals, full_y_errs.T, full_ordering):
            if start < x < end:
                subset_full_xs.append(x)
                subset_full_ys.append(y)
                subset_full_errs.append(err)
                subset_full_ordering.append(ordering)

        for x, y, err in zip(x_vals, y_vals, errs.T):
            if start < x < end:
                subset_xs.append(x)
                subset_ys.append(y)
                subset_errs.append(err)


        subset_full_errs = np.array(subset_full_errs).T
        subset_errs = np.array(subset_errs).T

        # prettify axes
        ax.set_xticks(ticks=[]) #10, 20, 27.5, 30.5], [2, 3, 4, 5])
        ax.xaxis.set_label_coords(.5, -0.05)
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.tick_params(axis='y', which='both', left=False)
        ax.axis([start, end, 0, 1.05])

        # plot everything
        ax.errorbar(subset_full_xs, subset_full_ys, yerr=subset_full_errs, fmt='.', ms=10,
                color='grey', capsize=0, lw=.3)
        ax.errorbar(subset_xs, subset_ys, yerr=subset_errs, fmt='.', ms=12,
                color='k', capsize=3, lw=2)
        ax = plot_pies_on_points(ax, subset_full_ordering, subset_full_xs,
                                 subset_full_ys, if_xlabel=True)
        if i == 0: ax = add_legend(ax)

    plt.savefig('figs/'+fig_filename, bbox_inches='tight')
    print('figs/'+fig_filename, fig.get_size_inches())

def plot_marginal_probs_for_each_diversity(sample_dict=import_fly_data(),
                                fig_filename='marginal_colonization_probabilities.pdf'):
    """Purpose: generate Fig. 3c of the paper, which shows the marginal
                probabilities of colonization for each species at each diversity 
       Inputs:  sample_dict ... as generated by import_fly_data(), either with
                'all' data or with some bootstrapped resampling
                fig_filename ... filename for the saved figure
       Returns: ax ... axes object with probability of colonization plot """

    # assemble data
    prob_dict = get_prob_outcomes(sample_dict)
    marginal_probs, marginal_prob_errs = get_marginal_probs_for_each_diversity(sample_dict)

    colors = ['#ff6300ff', '#ffd400ff', '#b02effff', '#009cf0ff', '#00cc0cff']

    fig, axs = plt.subplots(ncols=5, sharey=True)

    for ax_num,diversity in enumerate(range(1,6)):
        ax = axs[ax_num]

        for j,species in enumerate(range(5)):
            x_vals = []; y_vals = []; y_val_errs = []; full_ordering = []
            val = ''.join(['1' if i < diversity else '0' for i in range(5)])
            subval = (0,)
            full_ordering.append([val, subval])
            x_vals.append(diversity + (j-2)*0.15)
            y_vals.append(marginal_probs[diversity][species])
            y_val_errs.append(marginal_prob_errs[diversity][species])

            new_colors = np.copy(colors)
            new_colors[0] = new_colors[species]

            y_val_errs = np.array(y_val_errs).T

            ax.errorbar(x_vals, y_vals, yerr=y_val_errs, fmt='.', ms=12,
                    color=new_colors[0], capsize=3, lw=1)
            ax.axis([diversity-0.5, diversity+0.5, 0.0, 1.05])
            ax = plot_pies_on_points(ax, full_ordering, x_vals,
                                     y_vals, colors=new_colors,
                                     off_color='gray', my_circ_size=0.11,
                                     my_lw=1, fudge_factor=1.08)
            if ax_num == 0 and j == 0: ax = add_legend(ax, add_x=-0.3,
                                                       add_y=0.01)

            # prettify axes
            ax.set_xticks(ticks=[]) #10, 20, 27.5, 30.5], [2, 3, 4, 5])
            ax.xaxis.set_label_coords(.5, -0.05)
            ax.set_xlabel(diversity, fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=14)
            if ax_num > 0:
                ax.tick_params(axis='y', which='both', left=False)

    plt.subplots_adjust(wspace=0.08)
    fig.text(0.5, -0.02, r'\# species fed $N$', ha='center', fontsize=14)
    fig.text(0.008, 0.5, r'$p_N(i)$', va='center',
             rotation='vertical', fontsize=14)

    plt.savefig('figs/'+fig_filename, bbox_inches='tight')
    print('figs/'+fig_filename, fig.get_size_inches())

def plot_dendrogram(z, labels, ax, linewidth=0.5):
    """ Purpose: Plot dendrogram (as in Fig. 3d) based on clustering of z.
        Inputs: z ... matrix, rows will be clustered to produce dendrogram
                labels ... list of strings labeling each row
                ax ... ax to plot on
                linewidth ... float determining width of dendrogram lines
        Returns: new_ordering ... order successfully clustered dendrogram.
                                  dendrogram itself is implicitly plotted on
                                  ax. """
    def my_f(u, v):
        """ Purpose: Helper function, distance function for comparing rows of z
                     that ignores comparisions of NaN values.
            Inputs: u, v ... two length-5 lists (rows of z) to compare
            Output: total ... distance (as defined by my_f) between u and v """
        total = 0
        for uu, vv in zip(u, v):
            if np.isnan(uu) or np.isnan(vv):
                continue
            total += (uu - vv)**2
        total = np.sqrt(total)
        return total

    dendro_Z = hierarchy.linkage(z, method='complete', metric=my_f)

    # temporarily override the default line width to change dendrogram thickness:
    with plt.rc_context({'lines.linewidth': linewidth}):
        dendro_dict = hierarchy.dendrogram(dendro_Z, orientation='left',
                                           count_sort=True, labels=labels,
                                           no_labels=True, ax=ax,
                                           above_threshold_color='k',
                                           color_threshold=0)

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.axis('off')
    new_ordering = dendro_dict['ivl']
    return new_ordering

def plot_num_species_colonize(sample_dict=import_fly_data()):
    """ Purpose: Plot the average number of species that colonized as a
                 function of the number of species fed (Fig 3ab).
        Input: sample_dict ... as in other functions
        Output: void ... figure saved to pdf"""

    fig,(ax0,ax1) = plt.subplots(nrows=2, sharex=False, figsize=(3.6,9.2))

    num_species_colonize = [[] for i in range(5)]
    prop_species_colonize = [[] for i in range(5)]
    for key in sample_dict:
        if key == '00000': continue

        div = sum([1 for val in key if val == '1'])
        for sample in sample_dict[key]:
            num_species = sum([1 for val in sample if val > 0])
            num_species_colonize[div-1].append(num_species)
            prop_species_colonize[div-1].append(num_species/div)

    xs = []
    means = []
    lower_95s = []
    upper_95s = []
    for i,div in enumerate(num_species_colonize):
        xs.append(i+1)
        mean = np.mean(div)
        means.append(mean)
        lower_95, upper_95 = weightstats.DescrStatsW(div).tconfint_mean()
        lower_95s.append(mean - lower_95)
        upper_95s.append(upper_95 - mean)

    ax0.errorbar(xs, means, yerr=[lower_95s, upper_95s], color='k', fmt='.',
                 ms=18, capsize=4, lw=2.5)
    ax0.plot([i for i in range(1,6)], [i for i in range(1,6)], color='grey',
             ls='none', marker='^', ms=11)
    ax0.axis([None, None, 0, 5.5])
    ax0.set_xticks([i for i in range(1,6)])
    ax0.set_yticks([i for i in range(0,6)])
    ax0.tick_params(axis='both', which='major', labelsize=14)
    ax0.set_xlabel('\# species fed', fontsize=14)
    ax0.set_ylabel('\# species colonized', fontsize=14)

    xs = []
    means = []
    lower_95s = []
    upper_95s = []
    for i,div in enumerate(prop_species_colonize):
        xs.append(i+1)
        mean = np.mean(div)
        means.append(mean)
        lower_95, upper_95 = weightstats.DescrStatsW(div).tconfint_mean()
        lower_95s.append(mean - lower_95)
        upper_95s.append(upper_95 - mean)

    ax1.errorbar(xs, means, yerr=[lower_95s, upper_95s], color='k', fmt='.',
                 ms=18, capsize=4, lw=2.5)
    ax1.plot([i for i in range(1,6)], [1 for i in range(5)], ls='none',
             color='grey', marker='^', ms=11)
    ax1.set_xticks([i for i in range(1,6)])
    ax1.set_yticks([0, 0.5, 1])
    ax1.axis([None, None, 0, 1.05])
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_xlabel('\# species fed', fontsize=14)
    ax1.set_ylabel('proportion of species colonized', fontsize=14)

    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.1)
    plt.savefig('figs/num_and_prop_species_colonize_per_diversity.pdf', bbox_inches='tight')
    print('figs/num_and_prop_species_colonize_per_diversity.pdf', fig.get_size_inches())

def plot_model_comparisons(sample_dict=import_fly_data(), load_data=True, verbose=False):
    """ Demonstrate the ability of the independent and interaction models to
    predict the observed distributions of colonization outcomes (Fig 4).
    Inputs: load_data ... boolean whether to load data
            verbose ... boolean indicating whether to output diagnostic text
    Output: void ... figure saved as a pdf """

    prob_dict = get_prob_outcomes(sample_dict)

    # models with fixed probabilities of colonization
    model_labels, independent_probs = (
        get_independent_colonization_model_probs(verbose=False,
                                                 fit_data_w_diversity='all'))
    keys, pvals, log_likelihoods = (
        compute_statistics_of_independent_models(verbose=True,
                                                 load_data=load_data))

    # models with context-specific probabilities of colonization
    base_probs_all, interaction_all = get_interaction_params(
        fit_data_w_diversity='all', verbose=False, load_data=load_data)
    base_probs_2, interaction_2 = get_interaction_params(
        fit_data_w_diversity=2, verbose=False, load_data=load_data)
    keys_cf, pvals_cf, log_likelihoods_cf = (
        compute_statistics_of_interaction_models(
                use_data=[2, 'all'], verbose=True, load_data=True))

    model_labels = model_labels + ['interaction (div-2)', 'interaction (all)']

    colonize_matrix = []
    for probs in independent_probs:
        key_probs = []
        for key in keys:
            indices = [i for i,x in enumerate(key) if x=='1']
            key_model_prob = 1
            for idx in indices:
                key_model_prob = key_model_prob*probs[idx]
            full_subkey = tuple(range(len(indices)))
            observed_prob = prob_dict[key][full_subkey]
            key_probs.append(key_model_prob - observed_prob)
        colonize_matrix.append(key_probs)

    for probs,interaction in zip([base_probs_2, base_probs_all],
                                [interaction_2, interaction_all]):
        key_probs = []
        for key in keys:
            interaction_probs = get_interaction_probs(key, probs, interaction)
            indices = [i for i,x in enumerate(key) if x=='1']
            key_model_prob = 1
            for idx in indices:
                key_model_prob = key_model_prob*interaction_probs[idx]
            full_subkey = tuple(range(len(indices)))
            observed_prob = prob_dict[key][full_subkey]
            key_probs.append(key_model_prob - observed_prob)
        colonize_matrix.append(key_probs)

    pval_matrix = []
    for model_pvals in pvals:
        pval_matrix.append(model_pvals)
    for cf_pvals in pvals_cf:
        pval_matrix.append(cf_pvals)

    pval_matrix = np.array(pval_matrix)
    pval_matrix = np.array([[max(val, 1e-10) for val in row] for row in pval_matrix])

    if verbose:
        print(model_labels)
        for i,row in enumerate(pval_matrix.T):
            print(keys[i], [round(val, 3) for val in row])
        print(np.sum(pval_matrix > 0.05, axis=1))
        print(model_labels)

    xlabels = [' ' for i in range(len(keys))]
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    plt.subplots_adjust(hspace=-0.3)

    ax0 = plot_heatmap(colonize_matrix, ax0, xlabels=xlabels, ylabels=model_labels,
                       cmap='PiYG', cbar_label=r'$P_\text{M}(k) - P_\text{O}(k)$',
                       vbound=[-0.2, 0.2], fontsize=11, dashed=True,
                       cbararrow=True)
    cmap = plt.cm.get_cmap('Reds').reversed()
    ax1 = plot_heatmap(pval_matrix, ax1, xlabels=xlabels, ylabels=model_labels,
                       cmap=cmap, cbar_label=r'p-value',
                       vbound=[1e-3, .1], norm='log',
                       fontsize=11, dashed=True, cbararrow=True)

    key_list = []
    for key in keys:
        indices = [i for i,x in enumerate(key) if x=='1']
        key_list.append([key, tuple(range(len(indices)))])
    xvals = list([i+0.5 for i in range(len(keys))])
    yvals = [6.6 for i in range(len(keys))]

    ax0.set_xticks(ticks=[])
    ax0 = plot_pies_on_points(ax0, key_list, xvals, yvals, my_circ_size=0.45,
                              my_lw=0.5, fudge_factor=1.85)
    ax1.set_xticks(ticks=[])
    ax1 = plot_pies_on_points(ax1, key_list, xvals, yvals, my_circ_size=0.45,
                              my_lw=0.5, fudge_factor=1.85)
    ax0.text(-9.5, 0, r'\textbf{a)}', fontsize=16)
    ax1.text(-9.5, 0, r'\textbf{b)}', fontsize=16)


    plt.savefig('figs/comparison_of_model_predictions_log.pdf', bbox_inches='tight')
    print('figs/comparison_of_model_predictions_log.pdf', fig.get_size_inches())

def plot_clustered_marginal_probs(sample_dict=import_fly_data(), load_data=True, verbose=False):
    """ Plot a heatmap of the marginal probabilities of species in the presence
    of each other species (Fig 3d).
    Inputs: load_data ... boolean whether to load data
            verbose ... boolean indicating whether to output diagnostic text
    Output: void ... figure saved as a pdf """

    prob_dict = get_prob_outcomes(sample_dict)
    abcde = 'ABCDE'

    model_labels, independent_probs = (
        get_independent_colonization_model_probs(verbose=False,
                                                 fit_data_w_diversity='all'))
    keys, pvals, log_likelihoods = (
        compute_statistics_of_independent_models(verbose=False,
                                                 load_data=load_data))

    marg_prob_matrix = np.zeros((5, 5))
    marg_prob_residual_matrix = np.zeros((5, 5))
    num_colonized_matrix = np.zeros((5, 5))
    num_fed_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            num_fed = 0
            num_colonized = 0
            # consider only cases where both i and j were fed:
            for key in sample_dict:
                div = len([i for i in key if i == '1'])
                if key[i] == '1' and key[j] == '1':
                    for sample in sample_dict[key]:
                        num_fed += 1
                        if sample[i] > 0:
                            num_colonized += 1

            marg_prob_matrix[i,j] = num_colonized/num_fed
            num_colonized_matrix[i,j] = num_colonized
            num_fed_matrix[i,j] = num_fed

    for i in range(5):
        for j in range(5):
            marg_prob_residual_matrix[i,j] = (
                marg_prob_matrix[i,j] - marg_prob_matrix[i,i])
            if i == j:
                marg_prob_residual_matrix[i,j] = np.nan

    xlabels = [r'{}'.format(i) for i in range(5)]
    ylabels = [r'$\Delta p^{{\, j}}( \ {} \ )$'.format(i) for i in range(5)]

    fig, (ax0,ax1) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1,6]})
    plt.subplots_adjust(wspace=0.28)

    # plot data
    new_ordering = plot_dendrogram(marg_prob_residual_matrix, ylabels, ax0,
                                   linewidth=2.0)
    marg_prob_residual_matrix, ylabels = (
            reorder_vals(new_ordering, marg_prob_residual_matrix, ylabels))
    plot_heatmap(marg_prob_residual_matrix, ax1, xlabels=xlabels, ylabels=ylabels,
            cmap='PiYG', cbar_label=r'deviation in colonization odds $\Delta p^{\, j}(i)$',
            fontsize=14, ylabel='', xaxis_top=True, rotatey=True, vbound=[-.1, .1])

    # plot pies on y-axis
    key_list = []
    for key in keys[:5]:
        indices = [i for i,x in enumerate(key) if x=='1']
        key_list.append([key, tuple(range(len(indices)))])
    xvals = list([-0.455 for i in range(len(keys))])
    yvals = [i+0.53 for i in range(len(keys))]
    ax1 = plot_pies_on_points(ax1, key_list, xvals, yvals, my_circ_size=0.15,
                             my_lw=0.7, fudge_factor=0.8)

    # plot pies on x-axis
    key_list = []
    for key in keys[:5]:
        indices = [i for i,x in enumerate(key) if x=='1']
        key_list.append([key, tuple(range(len(indices)))])
    xvals = list([i+0.5 for i in range(len(keys))])
    yvals = [-0.3 for i in range(len(keys))]
    ax1 = plot_pies_on_points(ax1, key_list, xvals, yvals, my_circ_size=0.15,
                             my_lw=0.7, fudge_factor=0.8)

    ax1.text(2.45, -0.6, '$j$', fontsize=14)


    plt.savefig('figs/marginal_probs_w_different_species.pdf', bbox_inches='tight')
    print('figs/marginal_probs_w_different_species.pdf', fig.get_size_inches())

def plot_schematic_colonization_bar_plots(sample_dict=import_fly_data()):
    """ Plot barchart of colonization odds of an example combination ('10110'),
    as in Fig 1a.
    Output: void ... figure saved as a pdf """
    prob_dict = get_prob_outcomes(sample_dict)
    plot_keys = ['10110']
    labels = ['A', 'B', 'C', 'D', 'E']

    for key in plot_keys:
        x_vals = []
        y_vals = []

        idxs = [i for i,x in enumerate(key) if x == '1']
        num_samples = len(sample_dict[key])

        key_label = [labels[i] for i,val in enumerate(key) if val == '1']

        for i,subkey in enumerate(prob_dict[key]):
            xval_label = [labels[idxs[val]] for val in subkey]
            if xval_label:
                xval_label = i
            else:
                xval_label = i
            x_vals.append(xval_label)
            y_vals.append(prob_dict[key][subkey])

        x_vals = x_vals
        y_vals = y_vals[::-1]

        fig, ax = plt.subplots(figsize=(6.4, 1.2))
        barchart = ax.bar(x_vals, y_vals, color='k', edgecolor='k', lw=3)

        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

        ax.set_yticks([0, 0.5, 1])

        plt.tick_params(axis='x', which='both', bottom=False)

        ax.set_ylabel('P(outcome)')
        ax.set_xlabel('colonization outcome')
        ax.axis([None, None, 0, 1])

        # plot pies on x-axis
        key_list = []
        for subkey in prob_dict[key]:
            indices = [i for i,x in enumerate(key) if x=='1']
            key_list.append([key, subkey])
        key_list = key_list[::-1]

        xvals = list([i for i in range(len(key_list))])
        yvals = [-0.22 for i in range(len(key_list))]
        ax = plot_pies_on_points(ax, key_list, xvals, yvals, off_color='black', my_circ_size=0.28,
                                 my_lw=0.8, fudge_factor=1.0)

        plt.savefig('figs/schematic_colonization_bar_plots_{}_pies.pdf'.format(key),
                    bbox_inches='tight')
        plt.figure()


#############################################
### 
#############################################

if __name__ == '__main__':
    # PLOT FIG 2
    print('Plotting Fig. 2:')
    plot_all_colonization_probs()
    print()

    # PLOT FIG 3A AND 3B
    print('Plotting Fig. 3ab:')
    plot_num_species_colonize()
    # PLOT FIG 3C
    print('Plotting Fig. 3c:')
    plot_marginal_probs_for_each_diversity()
    # PLOT FIG 3D
    print('Plotting Fig. 3d:')
    plot_clustered_marginal_probs()
    print()

    # PLOT FIG 4
    print('Plotting Fig. 4:')
    plot_model_comparisons(load_data=True, verbose=False)
    print()

    # PRINT PROBABILITIES OF COLONIZATION OF EACH MODEL
    # AND EVALUATE EACH MODEL'S PERFORMANCE
    print('Evaluating model performances:')
    evaluate_model_performances()



    ### SUPPLEMENTARY CONTENT:

    ## PLOT COLONIZATION BAR PLOTS (as in subset of Fig. 1a)
    #print('Plotting Fig. 1a subset:')
    #plot_schematic_colonization_bar_plots()
