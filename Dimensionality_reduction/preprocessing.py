
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



def scaling_df(
    csv_file_name='HTRU_2.csv',
    dep_var='pulsar',
    names=['mean_ip', 'sd_ip', 'ec_ip', 
           'sw_ip', 'mean_dm', 'sd_dm', 
           'ec_dm', 'sw_dm', 'pulsar']):
    """Create a dataframe and then scale it 
    for modelling

    Args:
        csv_file_name (str, optional): csv file name. Defaults to 'HTRU_2.csv'.
        dep_var (str, optional): dep_varialbe name . Defaults to 'pulsar'.
        names (list, optional): names of the columns.
         Defaults to ['mean_ip', 'sd_ip', 'ec_ip', 'sw_ip', 'mean_dm', 'sd_dm', 'ec_dm', 'sw_dm', 'pulsar'].

    Returns:
        _type_: array of 
    """

    if names:
        data = pd.read_csv(
            csv_file_name,
            names=names)
    else:
        data = pd.read_csv(
            csv_file_name,
        ) 


    features = data[[i for i in data.columns if i!=
    dep_var]]
    robust_data = RobustScaler().fit_transform(
        features
    )
    return robust_data

def pca_show(
    data_):

    """Visualizing the principle 
    component analysis of the data_
    """

    pca_all = PCA()
    pca_all.fit(
        data_
    )
    cum_var = (np.cumsum(pca_all.explained_variance_ratio_))
    n_comp = [i for i in range(1, pca_all.n_components_ + 1)]

    # plot cumulative variance
    ax = sns.pointplot(x=n_comp, y=cum_var)
    ax.set(
        xlabel='number of components',
        ylabel='cum explained variance')


def pca_of_data(
    data_,
    pandas_dataframe,
    dep_var,
    no_component=3):
    """fitting priniciple component 
    analysis of the data_

    Args:
        data_ (numpy array): after fitting the 
                             data for sklearn function
        no_component (_type_): number of component 
        of principle compnent
        pandas_dataframe:pd.DataFrame
    """

    pca_3 = PCA(no_component)
    pca_3.fit(data_)

    data_3pc = pca_3.transform(data_)
    
    # Render the 3D plot
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(data_3pc[:, 0],
               data_3pc[:, 1],
               data_3pc[:, 2], 
               c=pandas_dataframe[dep_var],
               cmap=plt.cm.Set1,
               edgecolor='k',
               s=25,
               label=pandas_dataframe[dep_var])

    ax.legend(
        ["non-pulsars"], 
        fontsize="large")

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st principal component")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd principal component")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd principal component")
    ax.w_zaxis.set_ticklabels([])

def pca_of_data_2(data_,
                  pandas_df,
                  dep_var,
                  no_of_components=2):
    # Instantiate PCA with 2 components
    pca_2 = PCA(no_of_components)

    # Fit and transform scaled data
    pca_2.fit(data_)
    data_2pc = pca_2.transform(data_)

    # Render the 2D plot
    ax = sns.scatterplot(x=data_2pc[:,0], 
                         y=data_2pc[:,1], 
                         hue=pandas_df[dep_var],
                         palette=sns.color_palette("muted", n_colors=2))

    ax.set(xlabel='1st principal component',
           ylabel='2nd principal component',
           title='First two PCA directions')


