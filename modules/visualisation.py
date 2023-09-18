import copy
import os
import random
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from bertviz import head_view, model_view
from model import *
from clustering import Cluster
from tqdm import tqdm
import torch
from sklearn.manifold import TSNE
import umap as u
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as mcolors


class VisAtt(Model):
    def __init__(self, model_link, data_path, rank, project, tags, config, ddp, log):
        super().__init__(model_link, data_path, rank, project, tags, config, ddp, log)

    def getAttWeights(self, smile: str):
        inputs = self.tokenizer.encode_plus(smile, return_tensors='pt')
        outputs = self.model(inputs.input_ids.to(self.rank))
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        # 3 hidden_layers
        # each hidden layer has 12 attention heads - average for each hidden_layer
        return (tokens, [torch.mean(layer[0], dim=0) for layer in outputs.attentions])

    def buildAttDictionary(self, data):
        token_cols = {k: [] for k, v in self.tokenizer.vocab.items()}
        layer0, layer1, layer2 = copy.deepcopy(token_cols), copy.deepcopy(
            token_cols), copy.deepcopy(token_cols)
        for s in tqdm(data['smiles']):
            tokens, aw = self.getAttWeights(s)
            for n, layer in enumerate(aw):
                for ind, token in enumerate(tokens):
                    mean_token = float(layer[:, ind].mean())
                    if n == 0:
                        layer0[token].append(mean_token)
                    elif n == 1:
                        layer1[token].append(mean_token)
                    elif n == 2:
                        layer2[token].append(mean_token)
        l0 = {k: np.mean(v) for k, v in layer0.items()}
        l1 = {k: np.mean(v) for k, v in layer1.items()}
        l2 = {k: np.mean(v) for k, v in layer2.items()}
        return l0, l1, l2

    def buildAttArray(self, data):
        l0, l1, l2 = [], [], []
        for s in tqdm(data['smiles']):
            tokens, aw = self.getAttWeights(s)
            for n, layer in enumerate(aw):
                l = pd.DataFrame(layer.detach().cpu(),
                                 index=tokens, columns=tokens)
                df = pd.DataFrame(None, index=self.tokenizer.vocab,
                                  columns=self.tokenizer.vocab)
                hm = l.groupby(l.columns, axis=1).mean(
                ).T.groupby(l.columns, axis=1).mean()
                for i in range(len(hm)):
                    for j in range(len(hm)):
                        # get row
                        row = hm.iloc(0)[i]
                        col = hm.iloc(1)[j]
                        val = col[i]
                        df[col.name][row.name] = val
                if n == 0:
                    l0.append(df.to_numpy())
                elif n == 1:
                    l1.append(df.to_numpy())
                elif n == 2:
                    l2.append(df.to_numpy())
        l0, l1, l2 = np.array(l0, dtype=np.float64), np.array(
            l1, dtype=np.float64), np.array(l2, dtype=np.float64)
        empty = np.empty((3, l0.shape[1], l0.shape[2]))
        for l_ind, layer in enumerate([l0, l1, l2]):
            for i in range(l0.shape[1]):
                for j in range(l0.shape[2]):
                    a = layer[:, i, j]
                    a = a[~np.isnan(a)]
                    if len(a) == 0:
                        a = None
                    else:
                        a = np.mean(a)
                    empty[l_ind, i, j] = a
        return empty

    def visualiseAttBarPlot(self, attention_weights: dict):
        df = pd.DataFrame(attention_weights).dropna(axis=1).dropna(axis=0)
        df = df.reindex(df.mean().sort_values(ascending=False).index, axis=1)

        fig, axs = plt.subplots(3)
        for n in range(len(df)):
            l = df.iloc[n]
            axs[n].bar(l.index, l, )
            axs[n].set_yticks([])
            if n != 2:
                axs[n].set_xticks([])
            axs[n].tick_params(labelrotation=90)
        fig.show()

    def visualiseAttHeatmap(self, attention_weights: np.array, layer: int, axs, path: str = None):
        df = pd.DataFrame(
            attention_weights[layer, :, :], columns=self.tokenizer.vocab, index=self.tokenizer.vocab)
        df = df.loc[~(df.isna()).all(axis=0)].loc[:, ~(df.isna()).all(axis=0)]
        hm = sns.heatmap(df, xticklabels=True, yticklabels=True, ax=axs)
        hm.set_facecolor('xkcd:grey')
        if path is not None:
            plt.savefig(path)
            plt.clf()
            plt.close()
        return hm

class VisLatent(Model, Cluster):
    def __init__(self, model_link, data_path, rank, project, tags, config, ddp, log):
        super().__init__(model_link, data_path, rank, project, tags, config, ddp, log)

    def loadHiddenStates(self, dir_path):
        path = dir_path + '/'
        with open(path + 'hidden_states.pkl', 'rb') as f:
            self.hidden_states = pickle.load(f)
        with open(path + 'hidden_labels.pkl', 'rb') as f:
            self.hidden_labels = pickle.load(f)

    def tsne_(self, arr, perplexity=50):
        tsne = TSNE(metric='euclidean', perplexity=perplexity,
                    n_iter=2000, n_iter_without_progress=500,)
        return tsne.fit_transform(arr)
    
    def umap_(self, arr, n_neighbors=18, min_dist=0.6):
        umap = u.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean', init='random', random_state=111)
        return umap.fit_transform(arr)
    
    def plotHiddenStates(self, dir_path):
        self.loadHiddenStates(dir_path)
        fig, axs = plt.subplots(len(self.hidden_states),
                                len(self.hidden_states[0]))
        fig.set_figheight(10)
        for y, k in enumerate(self.hidden_states.keys()):
            labels = self.hidden_labels[k]
            for x, layer in enumerate(self.hidden_states[k]):
                layer = layer.reshape(layer.shape[0], -1)
                X = self.tsne(layer, 30)
                axs[y, x].scatter(X[:, 0], X[:, 1], alpha=0.7,
                                  c=labels, s=8, cmap='seismic')
                axs[y, x].get_xaxis().set_ticks([])
                axs[y, x].get_yaxis().set_ticks([])
                axs[len(self.hidden_states)-1, x].set_xlabel(f'Layer {x}')
                axs[y, 0].set_ylabel(f'Epoch {k}')

    def get_hidden_states(self, smile):
        inputs = self.tokenizer.encode_plus(
            smile, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(inputs.input_ids.to(
                self.rank), output_hidden_states=True)
            hs = torch.stack(list(outputs.hidden_states[-3:])).cpu().numpy()
        del inputs, outputs
        return hs

    def appendHiddenStates(self, df):
        df = df.copy()
        hs = np.concatenate(df['smiles'].apply(
            lambda x: self.get_hidden_states(x)), axis=1)
        return hs

    def groupByDBSCAN(self, df):
        df = df.copy()
        fps = self.getFingerPrints(df['smiles'])
        # tsne_X = self.tsne(fps)
        umap_X = self.umap(fps)
        # db_tsne = self.dbscan(tsne_X, min_cluster_size=50,
        #                       cluster_selection_epsilon=10)
        db_umap = self.dbscan(umap_X, min_cluster_size=25, cluster_selection_epsilon=1) 
        # df['dbscan_tsne_group'] = db_tsne.labels_
        df['dbscan_umap_group'] = db_umap.labels_
        return df

    def clusterHiddenStates(self, hs):
        hs = hs.reshape(hs.shape[0], hs.shape[1], -1)
        hs_clust = []
        for layer in hs:
            X = self.umap_(layer)
            hs_clust.append(X)
        return hs_clust

        # cluster hidden states
        # take in df and cluster hidden states by layer

    def pltHiddenStates(self, df: pd.DataFrame):
        df = df.reset_index()
        fig, axs = plt.subplots(4, 4,
                                gridspec_kw={'width_ratios': [1,1,1,.05]})
        fig.set_figheight(10)
        fig.set_figwidth(10)
        df = self.groupByDBSCAN(df)
        for i in range(2):
            if i == 1:
                self.loadModel(
                    './models/model-20-01-2023-unique-leaf-1839/model_state_dict.pt')
            hs = self.appendHiddenStates(df)
            hs_clust = self.clusterHiddenStates(hs)
            for j, X in enumerate(hs_clust):
                axs[i*2, j].scatter(X[:, 0], X[:, 1], c=df['labels'], alpha=.6, cmap='seismic', s=1)
                axs[i*2+1, j].scatter(X[:, 0], X[:, 1], c=df['dbscan_umap_group'], alpha=.6, cmap='rainbow', s=1)
            for i in range(4): #rows
                for j in range(4): #cols
                    axs[i,j].get_xaxis().set_ticks([])
                    axs[i,j].get_yaxis().set_ticks([])
                    if i == 0 and j in [0,1,2]:
                        axs[i,j].set_title(f'Layer {j}')
                    if j == 0 and i in [0,1]:
                        axs[i,j].set_ylabel('Pre-trained')
                    elif j == 0:
                        axs[i,j].set_ylabel('Fine-tuned')
                    if j == 3:
                        if i in [0,2]:
                            norm = mcolors.Normalize(vmin=min(df['labels']), vmax=max(df['labels']))
                            m = cm.ScalarMappable(norm=norm,  cmap='seismic')
                            fig.colorbar(m, cax=axs[i,j])
                        else:
                            cmap = mpl.cm.rainbow
                            cmaplist = [cmap(i) for i in range(cmap.N)]
                            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                                'Custom cmap', cmaplist, cmap.N)
                            bounds = list(np.unique(df['dbscan_umap_group']))
                            bounds = bounds + [max(bounds) + 1]
                            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                            cb = mpl.colorbar.ColorbarBase(axs[i,j], cmap=cmap, norm=norm,
                                                      spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i',)
                            bounds[-1] = ''
                            cb.set_ticklabels(bounds)
        plt.savefig('./data/figures/latent_space/latent_space_clusters_v2.png')
        print('Plot saved!')

if __name__ == '__main__':
    project = 'sheepy-optim'
    tags = ['test_rewrite']
    model_link = 'DeepChem/ChemBERTa-10M-MTR'
    data_path = '/data/ben/bindingDB/processed_data/bindingDB_OPRK_HUMAN_P41145.csv'
    rank = 0
    config = {}

    v = VisLatent(model_link, data_path, rank, project,
                  tags, config, ddp=False, log=False)
    v.pltHiddenStates(v.data.df)
    
    # v = VisAtt(model_link, data_path, rank, project,
    #            tags, config, ddp=False, log=False)
    # df = v.data.df
    # fig, axs = plt.subplots(2,3)
    # fig.set_figwidth(12)
    # fig.set_figheight(10)
    # fig.subplots_adjust(wspace=.1, hspace=.1)
    # for i in range(2): #row
    #     if i == 1:
    #         v.loadModel('./models/model-20-01-2023-unique-leaf-1839/model_state_dict.pt')
    #     aw = v.buildAttArray(df)
    #     for j in range(3): #col
    #         vmin = aw.min()
    #         vmax = aw.max()
    #         df_ = pd.DataFrame(aw[j, :, :], columns=v.tokenizer.vocab, index=v.tokenizer.vocab)
    #         df_ = df_.loc[~(df_.isna()).all(axis=0)].loc[:, ~(df_.isna()).all(axis=0)]
    #         hm = sns.heatmap(df_, xticklabels=True, yticklabels=True, cbar=False, ax=axs[i,j])
    #         if i == 0:
    #             hm.set_xticklabels([])
    #         if j != 0:
    #             hm.set_yticklabels([])
    #         hm.set_facecolor('xkcd:black')
    # plt.savefig('./data/figures/attention_heatmaps/hm-aw-OPRK_2.png'); plt.show()

    # v.loadModel('./models/model-23-02-2023-generous-deluge-2177/model_state_dict.pt')
    # hs = v.get_hidden_states(v.data.df['smiles'].iloc[0])
    # df = v.appendHiddenStates(v.data.df)
    # fps = v.getFingerPrints(v.data.df['smiles'])
    # tsne_X = v.tsne(fps)
    # umap_X = v.umap(fps)
    # db = v.dbscan(tsne_X, min_cluster_size=50, cluster_selection_epsilon=10)
    # len(np.unique(db.labels_))
    # v.plotdbscan(db)
    # plt.scatter(tsne_X[:,0], tsne_X[:,1], c=v.data.df['labels'], alpha=.1, cmap='seismic')
    # plt.scatter(tsne_X[:,0], tsne_X[:,1], c=db.labels_, alpha=.1, cmap='plasma')

    # db = v.dbscan(umap_X, min_cluster_size=50, cluster_selection_epsilon=3)
    # len(np.unique(db.labels_))
    # v.plotdbscan(db)
    # plt.scatter(umap_X[:,0], umap_X[:,1], c=v.data.df['labels'], alpha=.1, cmap='seismic')
    # plt.scatter(umap_X[:,0], umap_X[:,1], c=db.labels_, alpha=.1, cmap='plasma')

    # df, t = v.groupByDBSCAN(v.data.df)

    # load in model
    # pass in all smiles, get hidden states for all smiles
    # cluster all smiles based on tanimoto similarity using umap or tsne
    # now each smiles has a dbscan group
    # cluster latent space based on euclidean distance
    # colour in latent space with binding affinity
    # colour in latent space with dbscan groups
    # are they similar?

    # v = VisAtt(model_link, data_path, rank, project, tags, config, ddp=False, log=False)
    # # v.loadModel('./models/model-23-02-2023-generous-deluge-2177/model_state_dict.pt')

    # aw = v.buildAttArray(v.data.df[:50])
    # v.visualiseAttHeatmap(aw, 0)

    # for i, path in enumerate(['hm-attentionWeights-pretrain_v2', 'hm-attentionWeights-finetune_v2']):
    #     v = VisAtt(model_link, data_path, rank, project, tags, config, ddp=False, log=False)
    #     if i == 1:
    #         v.loadModel('./models/model-23-02-2023-curious-haze-2071/model_state_dict.pt')
    #     # print([p for n,p in list(v.model.named_parameters()) if n == 'roberta.encoder.layer.2.attention.self.query.weight'])
    #     aw = v.buildAttArray(v.data.df)
    #     for i in range(3):
    #         v.visualiseAttHeatmap(aw, i, f'./data/figures/attention_heatmaps/{path}-layer{i}-drd2.png')

    # v = VisAtt(model_link, data_path, rank, project, tags, config, ddp=False, log=False)
    # v.loadModel('./models/model-23-02-2023-generous-deluge-2177/model_state_dict.pt')
    # v.model.get_output_embeddings()
