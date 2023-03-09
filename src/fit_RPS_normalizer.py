# this is code to fit a PCA with RPS normalizer
from matplotlib import gridspec
import pandas as pd
from skimage.color import lab2rgb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
import config


class RPS_normalizer():
    """normalizer class that can fit a 2 componenent PCA model and then transform all the a,b coordinates from images into 'pigmentaiton' values
    which correspond to the RPS values
    """

    def __init__(self, df: object) -> None:
        """instanciation call

        Args:
            df (pandas.dataframe): dataframe to csv with l,a,b values
        """
        
        log = logging.getLogger("RPS_normalizer")
        self.log = log
        self.n_comp = 2
        self.df = df
        self.out_csv = config.results_dir + "retinal_pigmentation_score.csv"

    def fit(self) -> None:
        """fit a,b values to PCA model
        """

        self.log.info("Fitting PCA with {}".format(self.n_comp))
        pca =PCA(n_components=self.n_comp)
        pca.fit(self.df[['a','b']].dropna().to_numpy())

        #check to see if the primary vector is positive - then invert it!
        if (pca.components_[0][0] > 0) and (pca.components_[0][1] > 0):
            self.log.info("inerverting PCA, previous pca eigenvector", pca.components_[0])
            pca.components_[0] = -pca.components_[0]
            self.log.info('new pca eigenvector', pca.components_[0])

        self.pca = pca

    def transform(self) -> None:
        """Transforms the a,b data into 'pigmentation' column of the dataframe using the eigenvector with greatest eigenvalue
        """

        self.df['pigmentation'] = self.df[['a', 'b']].dropna().apply(lambda x: self.pca.transform([[x.a, x.b]])[0][0], axis=1)
        self.df.to_csv(self.out_csv, index=False)
    
    def save_example(self) -> None:
        """saves a representatitve example of various RPS values with corresponding distribution plot of all RPS values. This
        can be used by people who want to see what the RPS value really represents from their dataset
        """

        def plot_vals(idx, ax):
            # allows to plot at a given ax object with required information from the df at a given index

            lab = self.df.loc[idx][['L', 'a', 'b']].values.tolist()
            rps = self.df.loc[idx].pigmentation
            name = self.df.loc[idx].Name

            rgb = lab2rgb(lab)

            ax.imshow(rgb.reshape((1,1,3)))
            ax.set_title("{}\na,b values: [{:.1f}, {:.1f}]\n RPS score {:.1f}".format(name.split('/')[-1],lab[1],lab[2], rps))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
 

        indexes = self.df.pigmentation.sort_values().dropna().index

        fig = plt.figure(figsize=(20,10))
        gs = gridspec.GridSpec(2,4)

        ax0 = fig.add_subplot(gs[0,0])
        plot_vals(indexes[1], ax0)

        ax1 = fig.add_subplot(gs[0,1])
        plot_vals(indexes[int(len(indexes) * 1/3 )], ax1)

        ax2 = fig.add_subplot(gs[0,2])
        plot_vals(indexes[int( len(indexes) * 2/3)], ax2)

        ax3 = fig.add_subplot(gs[0,3])
        plot_vals(indexes[-2], ax3)

        ax4 = fig.add_subplot(gs[1,:])
        ax4.hist(self.df['pigmentation'])
        ax4.set_title("Histogram of the RPS for all images in the dataset")
        ax4.set_xlabel("RPS score")
        ax4.set_ylabel("number of images")

        plt.suptitle("Representative median retinal background colors and their corresponding RPS", fontsize=24)

        plt.tight_layout()
        plt.savefig(config.results_dir + 'RPS_representative_images.png')

#df = pd.read_csv("/data/anand/color_fundus/manuscript_code/EPIC_cohort_RPS.csv")
#RIPPER = RPS_normalizer(df)
#RIPPER.save_example()
