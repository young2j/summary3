from statsmodels.compat.python import (lrange, iterkeys, iteritems, lzip,
                                       itervalues)
from functools import reduce                   
from collections import OrderedDict
import numpy as np
import pandas as pd
import datetime
import textwrap
# from .table import SimpleTable
# from .tableformatting import fmt_latex, fmt_txt
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_latex, fmt_txt


class Summary(object):
    def __init__(self):
        self.tables = []
        self.settings = []
        self.extra_txt = []
        self.title = None

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        '''Display as HTML in IPython notebook.'''
        return self.as_html()

    def add_df(self, df, index=True, header=True, float_format='%.4f',
               align='r'):
        '''Add the contents of a DataFrame to summary table
        Parameters
        ----------
        df : DataFrame
        header: bool
            Reproduce the DataFrame column labels in summary table
        index: bool
            Reproduce the DataFrame row labels in summary table
        float_format: string
            Formatting to float data columns
        align : string
            Data alignment (l/c/r)
        '''

        settings = {'index': index, 'header': header,
                    'float_format': float_format, 'align': align}
        self.tables.append(df)
        self.settings.append(settings)

    def add_array(self, array, align='r', float_format="%.4f"):
        '''Add the contents of a Numpy array to summary table
        Parameters
        ----------
        array : numpy array (2D)
        float_format: string
            Formatting to array if type is float
        align : string
            Data alignment (l/c/r)
        '''

        table = pd.DataFrame(array)
        self.add_df(table, index=False, header=False,
                    float_format=float_format, align=align)

    def add_dict(self, d, ncols=2, align='l', float_format="%.4f"):
        '''Add the contents of a Dict to summary table
        Parameters
        ----------
        d : dict
            Keys and values are automatically coerced to strings with str().
            Users are encouraged to format them before using add_dict.
        ncols: int
            Number of columns of the output table
        align : string
            Data alignment (l/c/r)
        '''

        keys = [_formatter(x, float_format) for x in iterkeys(d)]
        vals = [_formatter(x, float_format) for x in itervalues(d)]
        data = np.array(lzip(keys, vals))

        if data.shape[0] % ncols != 0:
            pad = ncols - (data.shape[0] % ncols)
            data = np.vstack([data, np.array(pad * [['', '']])])

        data = np.split(data, ncols)
        data = reduce(lambda x, y: np.hstack([x, y]), data)
        self.add_array(data, align=align)

    def add_text(self, string):
        '''Append a note to the bottom of the summary table. In ASCII tables,
        the note will be wrapped to table width. Notes are not indendented.
        '''
        self.extra_txt.append(string)

    def add_title(self, title=None, results=None):
        '''Insert a title on top of the summary table. If a string is provided
        in the title argument, that string is printed. If no title string is
        provided but a results instance is provided, statsmodels attempts
        to construct a useful title automatically.
        '''
        if isinstance(title, str):
            self.title = title
        else:
            try:
                model = results.model.__class__.__name__
                if model in _model_types:
                    model = _model_types[model]
                self.title = 'Results: ' + model
            except:
                self.title = ''

    def add_base(self, results, alpha=0.05, float_format="%.4f", title=None,
                 xname=None, yname=None):
        '''Try to construct a basic summary instance.
        Parameters
        ----------
        results : Model results instance
        alpha : float
            significance level for the confidence intervals (optional)
        float_formatting: string
            Float formatting for summary of parameters (optional)
        title : string
            Title of the summary table (optional)
        xname : List of strings of length equal to the number of parameters
            Names of the independent variables (optional)
        yname : string
            Name of the dependent variable (optional)
        '''

        param = summary_params(results, alpha=alpha, use_t=results.use_t)
        info = summary_model(results)
        if xname is not None:
            param.index = xname
        if yname is not None:
            info['Dependent Variable:'] = yname
        self.add_dict(info, align='l')
        self.add_df(param, float_format=float_format)
        self.add_title(title=title, results=results)

    def as_text(self):
        '''Generate ASCII Summary Table
        '''

        tables = self.tables
        settings = self.settings
        title = self.title
        extra_txt = self.extra_txt

        pad_col, pad_index, widest = _measure_tables(tables, settings)

        rule_equal = widest * '='
        #TODO: this isn't used anywhere?
        rule_dash = widest * '-'

        simple_tables = _simple_tables(tables, settings, pad_col, pad_index)
        tab = [x.as_text() for x in simple_tables]

        tab = '\n'.join(tab)
        tab = tab.split('\n')
        tab[0] = rule_equal
        tab.append(rule_equal)
        tab = '\n'.join(tab)

        if title is not None:
            title = title
            if len(title) < widest:
                title = ' ' * int(widest/2 - len(title)/2) + title
        else:
            title = ''

        txt = [textwrap.wrap(x, widest) for x in extra_txt]
        txt = ['\n'.join(x) for x in txt]
        txt = '\n'.join(txt)

        out = '\n'.join([title, tab, txt])

        return out

    def as_html(self):
        '''Generate HTML Summary Table
        '''

        tables = self.tables
        settings = self.settings
        #TODO: this isn't used anywhere
        title = self.title

        simple_tables = _simple_tables(tables, settings)
        tab = [x.as_html() for x in simple_tables]
        tab = '\n'.join(tab)

        return tab

    def as_latex(self):
        '''Generate LaTeX Summary Table
        '''
        tables = self.tables
        settings = self.settings
        title = self.title
        if title is not None:
            title = '\\caption{' + title + '} \\\\'
        else:
            title = '\\caption{}'

        simple_tables = _simple_tables(tables, settings)
        tab = [x.as_latex_tabular() for x in simple_tables]
        tab = '\n\\hline\n'.join(tab)

        out = '\\begin{table}', title, tab, '\\end{table}'
        out = '\n'.join(out)
        return out
    # I added the output method based on pd.DataFrame().to_excel/csv(). 
    # I merged  the results when they are output, mainly in order to make 
    #  the output be distinguishable and more beautiful when printing. 
    #  Maybe there are other better ways.
    def to_excel(self,path=None):
        tables = self.tables
        import  os
        cwd = os.getcwd()
        if path:
            path = path 
        else:
            path = cwd + '\\summary_results.xlsx'
        summ_df = pd.concat(tables,axis=0)
        return summ_df.to_excel(path)
    def to_csv(self,path=None):
        tables = self.tables
        import  os
        cwd = os.getcwd()
        if  path:
            path = path 
        else:
            path = cwd + '\\summary_results.csv'
        summ_df = pd.concat(tables,axis=0)
        return summ_df.to_csv(path)

def _measure_tables(tables, settings):
    '''Compare width of ascii tables in a list and calculate padding values.
    We add space to each col_sep to get us as close as possible to the
    width of the largest table. Then, we add a few spaces to the first
    column to pad the rest.
    '''

    simple_tables = _simple_tables(tables, settings)
    
    #Bug1: If tables or settings is an empty list, 
             # then _simple_tables() will return [].
             # that means length is also empty , 
             # so max() will raise an error. 
    #Bug2: If table[i] just has one column, '/nsep' will raise ZeroDivisionError. 
             # So I added exception capture codes as follows.

    if simple_tables == []:
        len_max = 0
        pad_sep = None
        pad_index = None
    else:
        tab = [x.as_text() for x in simple_tables]
        length = [len(x.splitlines()[0]) for x in tab]
        len_max = max(length)
        pad_sep = []
        pad_index = []
        for i in range(len(tab)):
            nsep = tables[i].shape[1] - 1
            # I added the 'except' codes as follows  because nsep may be zero
            try:
                pad = int((len_max - length[i]) / nsep)
            except (ZeroDivisionError):
                pad = int((len_max - length[i]))
            pad_sep.append(pad)
            len_new = length[i] + nsep * pad
            pad_index.append(len_max - len_new)

    return pad_sep, pad_index, len_max


# Useful stuff
_model_types = {'OLS' : 'Ordinary least squares',
                'GLS' : 'Generalized least squares',
                'GLSAR' : 'Generalized least squares with AR(p)',
                'WLS' : 'Weigthed least squares',
                'RLM' : 'Robust linear model',
                'NBin': 'Negative binomial model',
                'GLM' : 'Generalized linear model'
                }


def summary_model(results):
    '''Create a dict with information about the model
    '''
    def time_now(*args, **kwds):
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d %H:%M')
    info = OrderedDict()

    # I added some informations of  Panel regression from the package linearmodels. 
    # Panel regression has some different attribute names, but it doesn't matter here.
    info['Model:'] = lambda x: x.model.__class__.__name__
    info['Model Family:'] = lambda x: x.family.__class.__name__
    info['Link Function:'] = lambda x: x.family.link.__class__.__name__
    info['Dependent Variable:'] = lambda x: x.model.endog_names
    # 1
    info['Dependent Variable:'] = lambda x: x.model.dependent.vars[0]

    info['Date:'] = time_now
    info['No. Observations:'] = lambda x: "%#6d" % x.nobs
    info['Df Model:'] = lambda x: "%#6d" % x.df_model
    info['Df Residuals:'] = lambda x: "%#6d" % x.df_resid
    info['Converged:'] = lambda x: x.mle_retvals['converged']
    info['No. Iterations:'] = lambda x: x.mle_retvals['iterations']
    info['Method:'] = lambda x: x.method
    info['Norm:'] = lambda x: x.fit_options['norm']
    info['Scale Est.:'] = lambda x: x.fit_options['scale_est']
    info['Cov. Type:'] = lambda x: x.fit_options['cov']
    # 2
    #I add the x.cov_type because there is no attribute fit_options like OLS model
    info['Covariance Type:'] = lambda x: x.cov_type
    info['Covariance Type:'] = lambda x: x._cov_type

    info['R-squared:'] = lambda x: "%#8.3f" % x.rsquared
    info['Adj. R-squared:'] = lambda x: "%#8.3f" % x.rsquared_adj
    info['Pseudo R-squared:'] = lambda x: "%#8.3f" % x.prsquared
    info['AIC:'] = lambda x: "%8.4f" % x.aic
    info['BIC:'] = lambda x: "%8.4f" % x.bic
    info['Log-Likelihood:'] = lambda x: "%#8.5g" % x.llf
    # 3
    info['Log-Likelihood:'] = lambda x: "%#8.5g" % x.loglike

    info['LL-Null:'] = lambda x: "%#8.5g" % x.llnull
    info['LLR p-value:'] = lambda x: "%#8.5g" % x.llr_pvalue
    info['Deviance:'] = lambda x: "%#8.5g" % x.deviance
    info['Pearson chi2:'] = lambda x: "%#6.3g" % x.pearson_chi2
    info['F-statistic:'] = lambda x: "%#8.4g" % x.fvalue
    # 4
    info['F-statistic:'] = lambda x: "%#8.4g" % x.f_statistic.stat

    info['Prob (F-statistic):'] = lambda x: "%#6.3g" % x.f_pvalue
    # 5
    info['Prob (F-statistic):'] = lambda x: "%#6.3g" % x.f_statistic.pval

    info['Scale:'] = lambda x: "%#8.5g" % x.scale
    # 6 
    info['Effects:'] = lambda x: ','.join(['%#8s' % i for i in x.included_effects])
   
    out = OrderedDict()
    for key, func in iteritems(info):
        try:
            out[key] = func(results)
        # NOTE: some models don't have loglike defined (RLM), so that's NIE
        except (AttributeError, KeyError, NotImplementedError):
            pass
    return out


def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, float_format="%.4f"):
    '''create a summary table of parameters from results instance
    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : string or None
        optional name for the endogenous variable, default is "y"
    xname : list of strings or None
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.
    float_format : string
        float formatting options (e.g. ".3g")
    Returns
    -------
    params_table : SimpleTable instance
    '''
    from linearmodels.panel.results import PanelEffectsResults
    from linearmodels.panel.results import RandomEffectsResults 
    from linearmodels.panel.results import PanelResults

    res_tuple = (PanelEffectsResults,PanelResults,RandomEffectsResults)

    if isinstance(results, tuple):
        results, params, std_err, tvalues, pvalues, conf_int = results
    # else:
    #     params = results.params
    #     bse = results.bse
    #     tvalues = results.tvalues
    #     pvalues = results.pvalues
    #     conf_int = results.conf_int(alpha)
   
   # I added Panel results whose some attributes name are different.
   # So I modified the code as follows.

    elif isinstance(results,res_tuple):
        bse = results.std_errors
        tvalues = results.tstats
        conf_int = results.conf_int(1-alpha)
    else:
        bse = results.bse
        tvalues = results.tvalues
        conf_int = results.conf_int(alpha) 
    params = results.params
    pvalues = results.pvalues

    data = np.array([params, bse, tvalues, pvalues]).T
    data = np.hstack([data, conf_int])
    data = pd.DataFrame(data)

    if use_t:
        data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']
    else:
        data.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']

    if not xname:
        # data.index = results.model.exog_names
        try:
            data.index = results.model.exog_names
        except (AttributeError):
            data.index = results.model.exog.vars
    else:
        data.index = xname

    return data

# The following function just can stack standard errors,but  we
    #  usually use t statistics in reality. I modified the function to 
    # support one of standard errors, t or pvalues by parameter 'show'.

    # Bug: There exists different names for intercept item in different models,
    # for example, an OLS model named it 'Intercept' while 'const' in logit models.
    # So I also added a function to uniform the name to facilitate the data merge.

## Vertical summary instance for multiple models
# def _col_params(result, float_format='%.4f', stars=True):
#     '''Stack coefficients and standard errors in single column
#     '''

#     # Extract parameters
#     res = summary_params(result)
#     # Format float
#     for col in res.columns[:2]:
#         res[col] = res[col].apply(lambda x: float_format % x)
#     # Std.Errors in parentheses
#     res.ix[:, 1] = '(' + res.ix[:, 1] + ')'
#     # Significance stars
#     if stars:
#         idx = res.ix[:, 3] < .1
#         res.ix[idx, 0] = res.ix[idx, 0] + '*'
#         idx = res.ix[:, 3] < .05
#         res.ix[idx, 0] = res.ix[idx, 0] + '*'
#         idx = res.ix[:, 3] < .01
#         res.ix[idx, 0] = res.ix[idx, 0] + '*'
#     # Stack Coefs and Std.Errors
#     res = res.ix[:, :2]
#     res = res.stack()
#     res = pd.DataFrame(res)
#     res.columns = [str(result.model.endog_names)]
def _col_params(result, float_format='%.4f', stars=True,show='t'):
    '''Stack coefficients and standard errors in single column
    '''
      #I add the parameter 'show' equals 't' to display tvalues by default,
      #'p' for pvalues and 'se' for std.err are alternative.
    
    # Extract parameters
    res = summary_params(result)
    # Format float
    # Note that scientific number will be formatted to 'str' type though '%.4f'
    for col in res.columns[:3]:
        res[col] = res[col].apply(lambda x: float_format % x)
    res.iloc[:,3] = np.around(res.iloc[:,3],4)
    
    # Significance stars
    # .ix method will be depreciated,so .loc is used.
    if stars:
        idx = res.iloc[:, 3] < .1
        res.loc[res.index[idx], res.columns[0]] += '*'
        idx = res.iloc[:, 3] < .05
        res.loc[res.index[idx], res.columns[0]] += '*'
        idx = res.iloc[:, 3] < .01
        res.loc[res.index[idx], res.columns[0]] += '*'
    # Std.Errors or tvalues or  pvalues in parentheses
    res.iloc[:,3] = res.iloc[:,3].apply(lambda x: float_format % x) # pvalues to str
    res.iloc[:, 1] = '(' + res.iloc[:, 1] + ')'
    res.iloc[:, 2] = '(' + res.iloc[:, 2] + ')'
    res.iloc[:, 3] = '(' + res.iloc[:, 3] + ')'

    # Stack Coefs and Std.Errors or pvalues
    if show is 't':
        res = res.iloc[:,[0,2]]
    elif show is 'se':
        res = res.iloc[:, :2]
    elif show is 'p':
        res = res.iloc[:,[0,3]]
    res = res.stack()
    res = pd.DataFrame(res)
    try:
        res.columns = [str(result.model.endog_names)]
    except (AttributeError):
        res.columns = result.model.dependent.vars
    
    
    # I added the index name transfromation function 
    # to deal with MultiIndex and single level index.
    def _Intercept_2const(df):
        from pandas.core.indexes.multi import MultiIndex
        if isinstance(df.index, MultiIndex):
            new_index = []
            for v in df.index.values:
                v = list(v)
                if 'Intercept' in v:
                    v[v.index('Intercept')] = 'const'
                new_index.append(v)
            multi_index = lzip(*new_index)
            df.index = MultiIndex.from_arrays(multi_index)            
        else:
            index_value = df.index.tolist()
            if 'Intercept' in index_value:
                index_value[index_value.index('Intercept')] = 'const'
            df.index = index_value
        return df
    return _Intercept_2const(res)



# def _col_info(result, info_dict=None):
#     '''Stack model info in a column
#     '''
#     if info_dict is None:
#         info_dict = {}
#     out = []
#     index = []
#     for i in info_dict:
#         if isinstance(info_dict[i], dict):
#             # this is a specific model info_dict, but not for this result...
#             continue
#         try:
#             out.append(info_dict[i](result))
#         except:
#             out.append('')
#         index.append(i)
#     out = pd.DataFrame({str(result.model.endog_names): out}, index=index)
#     return out
   
   # I modified the above function,main work is that 
   # I rename the parameter 'info_dict' to 'more_info',which is a list not a dict.
   # Besides, I build a default dict to contain some model information 
   # from summary_model(), that will be printed by default and 
   # users can append other statistics by more_info parameter.
def _col_info(result, more_info=None):
    '''Stack model info in a column
    '''
    model_info = summary_model(result)
    default_info_ = OrderedDict()
    default_info_['Model:'] = lambda x: x.get('Model:')
    default_info_['No. Observations:'] = lambda x: x.get('No. Observations:')
    default_info_['R-squared:'] = lambda x: x.get('R-squared:')
    default_info_['Adj. R-squared:'] = lambda x: x.get('Adj. R-squared:')                    
    default_info_['Pseudo R-squared:'] = lambda x: x.get('Pseudo R-squared:')
    default_info_['F-statistic:'] = lambda x: x.get('F-statistic:')
    default_info_['Covariance Type:'] = lambda x: x.get('Covariance Type:')
    default_info_['Eeffects:'] = lambda x: x.get('Effects:')
    default_info_['Covariance Type:'] = lambda x: x.get('Covariance Type:')

    default_info = default_info_.copy()
    for k,v in default_info_.items():
        if v(model_info):
            default_info[k] = v(model_info)
        else:
            default_info.pop(k) # pop the item whose value is none.
            
    if more_info is None:
        more_info = default_info
    else:
        if not isinstance(more_info,list):
            more_info = [more_info]
        for i in more_info:
            try:
                default_info[i] = getattr(result,i)
            except (AttributeError, KeyError, NotImplementedError) as e:
                raise e
        more_info = default_info
    try:
        out = pd.DataFrame(more_info, index=[result.model.endog_names]).T
    except (AttributeError):
        out = pd.DataFrame(more_info, index=result.model.dependent.vars).T
    return out

# def _make_unique(list_of_names):
#     if len(set(list_of_names)) == len(list_of_names):
#         return list_of_names
#     # pandas does not like it if multiple columns have the same names
#     from collections import defaultdict
#     name_counter = defaultdict(str)
#     header = []
#     for _name in list_of_names:
#         name_counter[_name] += "I"
#         header.append(_name+" " + name_counter[_name])
#     return header

   # Above function has a flaw that non-duplicated names will be add a suffix.
   # And the time when endog_names duplicate four or more times ,the y 
   # names will be like 'y IIII' or 'y IIIIII...'.So I used the Arabic numerals.
def _make_unique(list_of_names):
    if len(set(list_of_names)) == len(list_of_names):
        return list_of_names
    # pandas does not like it if multiple columns have the same names
    from collections import defaultdict
    dic_of_names = defaultdict(list)
    for i,v in enumerate(list_of_names):
        dic_of_names[v].append(i)
    for v in  dic_of_names.values():
        if len(v)>1:
            c = 0
            for i in v:
                c += 1
                list_of_names[i] += '_%i' % c
    return list_of_names

   # The following function is the most critical to work.
   # In this function  I added the parameters 'show' and 'title',
   # and changed the default value of 'stars' into 'True',
   # Then  I changed the dict parameter 'info_dict' as a list one named 'more_info'.
   # Finally I put 'const'  at the first location by default in regressor_order.

   # Bug: np.unique() will disrupt the original order of list,
   # this can result in index confusion.

# def summary_col(results, float_format='%.4f', model_names=[], stars=False,
#                 info_dict=None, regressor_order=[]): 
def summary_col(results, float_format='%.4f', model_names=[], stars=True,
                more_info=None, regressor_order=[],show='t',title=None): 
    # I added the parameter 'show' and changed the default of 'stars' into 'True',
    # then renamed the dict parameter 'info_dict' as a list one 'more_info'
    # finally assigned the regressor_order a initial value ['const']
    """
    Summarize multiple results instances side-by-side (coefs and SEs)
    Parameters
    ----------
    results : statsmodels results instance or list of result instances
    float_format : string
        float format for coefficients and standard errors
        Default : '%.4f'
    model_names : list of strings of length len(results) if the names are not
        unique, a roman number will be appended to all model names
    stars : bool
        print significance stars
    info_dict : dict
        dict of lambda functions to be applied to results instances to retrieve
        model info. To use specific information for different models, add a
        (nested) info_dict with model name as the key.
        Example: `info_dict = {"N":..., "R2": ..., "OLS":{"R2":...}}` would
        only show `R2` for OLS regression models, but additionally `N` for
        all other results.
        Default : None (use the info_dict specified in
        result.default_model_infos, if this property exists)
    regressor_order : list of strings
        list of names of the regressors in the desired order. All regressors
        not specified will be appended to the end of the list.
    """
    
    if not isinstance(results, list):
        results = [results]

    cols = [_col_params(x, stars=stars, float_format=float_format,show=show) for x in
            results]

    # Unique column names (pandas has problems merging otherwise)
    if model_names:
        colnames = _make_unique(model_names)
    else:
        colnames = _make_unique([x.columns[0] for x in cols])
    for i in range(len(cols)):
        cols[i].columns = [colnames[i]]

    merg = lambda x, y: x.merge(y, how='outer', right_index=True,
                                left_index=True)
    summ = reduce(merg, cols)

    # if regressor_order:
    if not regressor_order:
        regressor_order = ['const']
    
    varnames = summ.index.get_level_values(0).tolist()
    ordered = [x for x in regressor_order if x in varnames]
    unordered = [x for x in varnames if x not in regressor_order + ['']]
    
    # Note: np.unique can disrupt the original order  of list 'unordered'.
    # Then pd.Series().unique()  works well.
    
    # order = ordered + list(np.unique(unordered))
    order = ordered + list(pd.Series(unordered).unique())

    f = lambda idx: sum([[x + 'coef', x + 'stde'] for x in idx], [])
    # summ.index = f(np.unique(varnames))
    summ.index = f(pd.Series(varnames).unique())
    summ = summ.reindex(f(order))
    summ.index = [x[:-4] for x in summ.index]

    idx = pd.Series(lrange(summ.shape[0])) % 2 == 1
    summ.index = np.where(idx, '', summ.index.get_level_values(0))
    summ = summ.fillna('')
    
    # add infos about the models.
#     if info_dict:
#         cols = [_col_info(x, info_dict.get(x.model.__class__.__name__,
#                                            info_dict)) for x in results]
#     else:
#         cols = [_col_info(x, getattr(x, "default_model_infos", None)) for x in
#                 results]
    cols = [_col_info(x,more_info=more_info) for x in results]
    
    # use unique column names, otherwise the merge will not succeed
    for df , name in zip(cols, _make_unique([df.columns[0] for df in cols])):
        df.columns = [name]
    merg = lambda x, y: x.merge(y, how='outer', right_index=True,
                                left_index=True)
    info = reduce(merg, cols)
    info.columns = summ.columns
    info = info.fillna('')
#     dat = pd.DataFrame(np.vstack([summ, info]))  # pd.concat better, but error
#     dat.columns = summ.columns
#     dat.index = pd.Index(summ.index.tolist() + info.index.tolist())
#     summ = dat

#     summ = summ.fillna('')

#     smry = Summary()
#     smry.add_df(summ, header=True, align='l')
#     smry.add_text('Standard errors in parentheses.')
#     if stars:
#         smry.add_text('* p<.1, ** p<.05, ***p<.01')*p<.01')
#     return smry

    if show is 't':
        note = ['\t t statistics in parentheses.']
    if show is 'se':
        note = ['\t Std. error in parentheses.']
    if show is 'p':
        note = ['\t pvalues in parentheses.']
    if stars:
        note +=  ['\t * p<.1, ** p<.05, ***p<.01']
    #Here  I tried two ways to put extra text in index-location 
    # or columns-location,finally found the former is better.
#     note_df = pd.DataFrame(note,index=['note']+['']*(len(note)-1),columns=[summ.columns[0]])
    note_df = pd.DataFrame([ ],index=['note:']+note,columns=summ.columns).fillna('')
#     summ_all = pd.concat([summ,info,note_df],axis=0)
    
    # I construct a title DataFrame and adjust the location of title corresponding to the length of columns. 
    if title is not None:
        title = str(title)
    else:
        title = '\t Results Summary'
        
    # Here I tried to construct a title DataFrame and 
    # adjust the location of title corresponding to the length of columns. 
    # But I failed because of not good printing effect.

    # col_len = len(summ.columns)
    # fake_data = ['']*col_len
    # if col_len % 2 == 1:
    #     from math  import ceil
    #     i = ceil(col_len/2)
    # else:
    #     i = int(col_len/2)
    # fake_data[i-1] = title
    # title_df = pd.DataFrame([fake_data],index=[''],columns=summ.columns).fillna('')
    title_df = pd.DataFrame([],index=[title],columns=summ.columns).fillna('')
    
    smry = Summary()
    smry.add_df(title_df,header=False,align='l')
    smry.add_df(summ, header=True, align='l')
    smry.add_df(info, header=False, align='l')
    smry.add_df(note_df, header=False, align='l')
    return smry



def _formatter(element, float_format='%.4f'):
    try:
        out = float_format % element
    except:
        out = str(element)
    return out.strip()


def _df_to_simpletable(df, align='r', float_format="%.4f", header=True,
                       index=True, table_dec_above='-', table_dec_below=None,
                       header_dec_below='-', pad_col=0, pad_index=0):
    dat = df.copy()
    dat = dat.applymap(lambda x: _formatter(x, float_format))
    if header:
        headers = [str(x) for x in dat.columns.tolist()]
    else:
        headers = None
    if index:
        stubs = [str(x) + int(pad_index) * ' ' for x in dat.index.tolist()]
    else:
        dat.ix[:, 0] = [str(x) + int(pad_index) * ' ' for x in dat.ix[:, 0]]
        stubs = None
    st = SimpleTable(np.array(dat), headers=headers, stubs=stubs,
                     ltx_fmt=fmt_latex, txt_fmt=fmt_txt)
    st.output_formats['latex']['data_aligns'] = align
    st.output_formats['txt']['data_aligns'] = align
    st.output_formats['txt']['table_dec_above'] = table_dec_above
    st.output_formats['txt']['table_dec_below'] = table_dec_below
    st.output_formats['txt']['header_dec_below'] = header_dec_below
    st.output_formats['txt']['colsep'] = ' ' * int(pad_col + 1)
    return st


def _simple_tables(tables, settings, pad_col=None, pad_index=None):
    simple_tables = []
    float_format = '%.4f'
    if pad_col is None:
        pad_col = [0] * len(tables)
    if pad_index is None:
        pad_index = [0] * len(tables)
    for i, v in enumerate(tables):
        index = settings[i]['index']
        header = settings[i]['header']
        align = settings[i]['align']
        simple_tables.append(_df_to_simpletable(v, align=align,
                                                float_format=float_format,
                                                header=header, index=index,
                                                pad_col=pad_col[i],
                                                pad_index=pad_index[i]))
    return simple_tables