from data import DataManager
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.ML.Cluster import Butina
from sklearn.manifold import TSNE
from rdkit import DataStructs
import seaborn as sns
import matplotlib.pyplot as plt
import umap as umap_
from umap import plot as umap_plot
import hdbscan

import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors as mcolors

class Cluster():
    def getFingerPrints(self, smiles):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
        return fps
    
    def getPairwiseTanimoto(self, fps):            
        return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])

    def tanimoto_dist(self, a,b):
        dotprod = np.dot(a,b)
        tc = dotprod / (np.sum(a) + np.sum(b) - dotprod)
        return 1.0-tc
    
    def tsne(self, fps, perplexity=50):
        fps = np.array(fps)
        tsne = TSNE(metric=self.tanimoto_dist, perplexity=perplexity, n_iter=2000, n_iter_without_progress=500)
        return tsne.fit_transform(fps)
    
    def draw_tsne(self, fps, col_labels=None, perplexity=50):
        t = self.tsne(fps, perplexity)
        # title = f'tsne_plt-perplex{perplexity}'
        title = 'tsne-tanimoto_similarity-OPRK'
        plt.scatter(t[:,0], t[:,1], c=col_labels, cmap='seismic', alpha=.6, s=5)
        norm = mcolors.Normalize(vmin=min(col_labels), vmax=max(col_labels))
        m = cm.ScalarMappable(norm=norm,  cmap='seismic')
        plt.colorbar(m)
        # plt.title(title, fontsize=12)
        plt.savefig('./data/figures/tsne/' + title + '.png')
        plt.show(); plt.close()
        print(title + ' saved.')
        
    def tsne_gridSearch(self, fps, col_labels):
        for perpl in range(10,100, 10):
            self.draw_tsne(fps, col_labels, perpl)
            
    def umap(self, fps, n_neighbors=18, min_dist=0.6):
        fps = np.array(fps)
        umap = umap_.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=self.tanimoto_dist, random_state=444)
        return umap.fit_transform(fps)
        
    def draw_umap(self, fps, n_neighbors=18, min_dist=.6, col_labels=None):
        u = self.umap(fps, n_neighbors, min_dist)
        # title = f'umap_plt-nn{n_neighbors}-mindist{min_dist}'
        title = 'umap-tanimoto_similarity-OPRK'
        plt.scatter(u[:,0], u[:,1], c=col_labels, cmap='seismic', alpha=.6, s=5)
        norm = mcolors.Normalize(vmin=min(col_labels), vmax=max(col_labels))
        m = cm.ScalarMappable(norm=norm,  cmap='seismic')
        plt.colorbar(m)
        plt.savefig('./data/figures/umap/' + title + '.png')
        plt.show(); plt.close()
        print(title + ' saved.')
        
    def umap_gridSearch(self, fps, col_labels):
        for n_neighbour in range(2,20,2):
            for min_dist in np.linspace(0,1, 11):
                self.draw_umap(fps, col_labels, n_neighbour, min_dist)
                
    def dbscan(self, X, min_cluster_size, cluster_selection_epsilon):
        cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon ,gen_min_span_tree=True)
        cluster.fit(X)
        return cluster
        
    def plotdbscan(self, cluster: hdbscan.HDBSCAN):
        cluster.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                            edge_alpha=0.6,
                                            node_size=90,
                                            edge_linewidth=2)
        
    def plotClusterMap(self, data):
        fps = c.getFingerPrints(data['smiles'])
        
        cols = data['labels'].to_numpy()
        minima = min(cols)
        maxima = max(cols)

        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='plasma')
        cols = mapper.to_rgba(cols)
        pt = self.getPairwiseTanimoto(fps)
        fig = sns.clustermap(pt, xticklabels=False,  yticklabels=False, cmap='fire', row_colors=cols,)
        
        divider = make_axes_locatable(fig.ax_cbar)
        cax2 = divider.append_axes("right", pad="500%", size='100%')
        norm = matplotlib.colors.Normalize(vmin=min(data['labels']), vmax=max(data['labels']), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='plasma')
        fig.fig.colorbar(mapper, cax=cax2, orientation="vertical")
        plt.savefig('./data/figures/heatmaps/clustermap-tanimoto_similarity_OPRK.png')

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-10M-MTR')
    d = DataManager('/data/ben/bindingDB/processed_data/bindingDB_OPRK_HUMAN_P41145.csv', tok) 
    #d = DataManager('/data/ben/bindingDB/processed_data/bindingDB_DRD2_HUMAN_P14416.csv', tok)
    c = Cluster()
    # c.plotClusterMap(d.df)
    df = d.df
    fps = c.getFingerPrints(df['smiles'])
    # c.draw_umap(fps, col_labels=df['labels'])
    c.draw_tsne(fps, col_labels=df['labels'])