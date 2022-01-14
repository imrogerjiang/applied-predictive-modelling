import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.stats import norm

def qq_plot(x):
    """
    Plots against a standard normal distribution.
    
    Parameters
    ----------
    arg1 : array like
        observations to test against a normal distribution
    
    Returns
    ----------
    None
    """
       
    n = len(x)
    x = pd.Series(x)
    sample = ((x - x.mean())/x.std()).sort_values(ascending=True)
    theoretical = [norm.ppf((k+0.5)/n) for k in range(n)]
    plt.scatter(x=theoretical,y=sample)
    plt.axline((0,0), slope=1, color="darksalmon")
    plt.show()
    
def boxcox_transform(df, lambdas=None):
    """
    Applies the boxcox transformation to all variables in input data frame.
    
    If lambdas are specified, they are used for the boxcox transformation. If they aren't specified MLE is used to estimate them.
    
    Parameters:
        arg1 (pd.DataFrame): data frame to be transformed
        arg2 (array like, optional): lambdas to be applied for boxcox transformation.
    
    Returns:
        pd.Dataframe: Box Cox transformed dataframe
        list: Lambdas parameters used in the Box Cox transformation
    """
    bc = pd.DataFrame()
    bc_lambdas = []
    for i, c in enumerate(df.columns):
        if lambdas is None:
            a, b = boxcox(df[c])
            bc_lambdas.append(b)
        else:
            a = boxcox(df[c], lmbda=lambdas[i])
        a = pd.Series(a, name=c, index=df.index)
        bc = pd.concat([bc, a], axis="columns")
    return bc, bc_lambdas

def pca(ds, boxcox=False, max_components=None):
    """
    Applies principle component regression to find R2 of test dataset.
    
    This function uses the train/val split to find the best number of components. It then fits a model on train_val using best number of components to predict the test dataset. Finally it returns the R2 of the test dataset.
    
    Parameters:
        arg1 (dict): dictionary of pd.DataFrames. Should contain {
            "x_train": predictor values of training set,
            "x_val": predictor values of validation set,
            "x_train_val": predictor values of training and validation sets,
            "x_test": predictor values of testing set,
            "y_train": target values of training set,
            "y_val": target values of validation set,
            "y_train_val": target values of training and validation sets,
            "y_test": target values of testing set
        }
        arg2 (bool): whether Box Cox transformation is to be applied to predictor variables.
        arg3 (int): maximum number of components used in pca
    
    Returns:
        float: R2 value from applying the model trained train_val on test.
    """
    pca_scores = []
    
    if boxcox:
        ds_bc = {}
        ds_bc["x_train"], lambdas = boxcox_transform(ds["x_train"])
        ds_bc["x_val"], _ = boxcox_transform(ds["x_val"], lambdas=lambdas)
        ds_bc["x_train_val"], lambdas = boxcox_transform(ds["x_train_val"])
        ds_bc["x_test"], _ = boxcox_transform(ds["x_test"], lambdas=lambdas)
    else:
        ds_bc = ds
    
    if max_components is None: max_components=min(ds_bc["x_train"].shape)-1
    for n_components in range(1, max_components):
        pca_x = {}
        pca = PCA(n_components=n_components)
        ols_pca = linear_model.LinearRegression()

        pca_x["train"] = pca.fit_transform(ds_bc["x_train"])
        pca_x["val"] = pca.transform(ds_bc["x_val"])
        pca_x["test"] = pca.transform(ds_bc["x_test"])

        ols_pca.fit(X=pca_x["train"], y=ds["y_train"])
        pca_scores.append(ols_pca.score(X=pca_x["val"], y=ds["y_val"]))

    n = pca_scores.index(max(pca_scores))+1
    pca_final = PCA(n_components=n)
    pca_x["train_val"] = pca_final.fit_transform(ds_bc["x_train_val"])
    pca_x["test"] = pca_final.transform(ds_bc["x_test"])
    ols_pca.fit(pca_x["train_val"], ds["y_train_val"])

    return ols_pca.score(pca_x["test"], ds["y_test"])

def pls(ds, boxcox=False, max_components=None):
    """
    Applies partial least squares to find R2 of test dataset.
    
    This function uses the train/val split to find the best number of components. It then fits a model on train_val using best number of components to predict the test dataset. Finally it returns the R2 of the test dataset.
    
    Parameters:
        arg1 (dict): dictionary of pd.DataFrames. Should contain {
            "x_train": predictor values of training set,
            "x_val": predictor values of validation set,
            "x_train_val": predictor values of training and validation sets,
            "x_test": predictor values of testing set,
            "y_train": target values of training set,
            "y_val": target values of validation set,
            "y_train_val": target values of training and validation sets,
            "y_test": target values of testing set
        }
        arg2 (bool): whether Box Cox transformation is to be applied to predictor variables.
        arg3 (int): maximum number of components used in pca
    
    Returns:
        float: R2 value from applying the model trained train_val on test.
    """
    pls_scores = []

    if boxcox:
        ds_bc = {}
        ds_bc["x_train"], lambdas = boxcox_transform(ds["x_train"])
        ds_bc["x_val"], _ = boxcox_transform(ds["x_val"], lambdas=lambdas)
        ds_bc["x_train_val"], lambdas = boxcox_transform(ds["x_train_val"])
        ds_bc["x_test"], _ = boxcox_transform(ds["x_test"], lambdas=lambdas)
    else:
        ds_bc = ds
    
    if max_components is None: max_components=min(ds_bc["x_train"].shape)-1
    for n_components in range(1, max_components):
        pls = PLSRegression(n_components=n_components)
        pls_x = {}
        pls.fit(ds_bc["x_train"], ds["y_train"])
        pls_scores.append(pls.score(X=ds_bc["x_val"], y=ds["y_val"]))

    # Select number of components with highest R2 score
    # Fit regression on training and validation set
    # Score on test set
    n = pls_scores.index(max(pls_scores))+1
    pls_final = PLSRegression(n_components=n)
    pls_final.fit(ds_bc["x_train_val"], ds["y_train_val"])
    return pls_final.score(X=ds_bc["x_test"], y=ds["y_test"])

def penalised_regression(ds, model, boxcox=False):
    """
    Applies penalised regression to find R2 of test dataset.
    
    This function uses the train/val split to find the best penalty parameter. It then fits a model on train_val using the best penalty parameter to predict the test dataset. Finally it returns the R2 of the test dataset.
    
    Parameters:
        arg1 (dict): dictionary of pd.DataFrames. Should contain {
            "x_train": predictor values of training set,
            "x_val": predictor values of validation set,
            "x_train_val": predictor values of training and validation sets,
            "x_test": predictor values of testing set,
            "y_train": target values of training set,
            "y_val": target values of validation set,
            "y_train_val": target values of training and validation sets,
            "y_test": target values of testing set
        }
        arg2 (function): penalised regression model to be used. Either sklearn.linear_model.Lasso or .Ridge
        arg3 (bool): whether Box Cox transformation is to be applied to predictor variables.
    
    Returns:
        float: R2 value from applying the model trained train_val on test.
    """
    if boxcox:
        ds_bc = {}
        ds_bc["x_train"], lambdas = boxcox_transform(ds["x_train"])
        ds_bc["x_val"], _ = boxcox_transform(ds["x_val"], lambdas=lambdas)
        ds_bc["x_train_val"], lambdas = boxcox_transform(ds["x_train_val"])
        ds_bc["x_test"], _ = boxcox_transform(ds["x_test"], lambdas=lambdas)
    else:
        ds_bc = ds
    
    scores = []
    alphas = [10**(i/4) for i in range(-30, 0)]
    for alpha in alphas:
        penalised_reg = model(alpha=alpha, max_iter=10_000)
        penalised_reg.fit(ds_bc["x_train"], ds["y_train"])
        scores.append(penalised_reg.score(ds_bc["x_val"], ds["y_val"]))

    best_alpha = alphas[scores.index(max(scores))]
    final_model = model(alpha=best_alpha, max_iter=10_000)
    final_model.fit(ds_bc["x_train_val"], ds["y_train_val"])
    return final_model.score(ds_bc["x_test"], ds["y_test"])