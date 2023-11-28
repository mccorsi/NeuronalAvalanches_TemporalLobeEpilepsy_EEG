# TODO: https://towardsdatascience.com/chord-diagrams-of-protein-interaction-networks-in-python-9589affc8b91
# create a script from a matrix and a vector of labels - plot connectome
import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from nxviz.plots import CircosPlot
#%%
protein_list = ['TPH1','COMT','SLC18A2','HTR1B','HTR2C','HTR2A','MAOA',
            'TPH2','HTR1A','HTR7','SLC6A4','GABBR2','POMC','GNAI3',
            'NPY','ADCY1','PDYN','GRM2','GRM3','GABBR1']
proteins = '%0d'.join(protein_list)
url = 'https://string-db.org/api/tsv/network?identifiers=' + proteins + '&species=9606'
r = requests.get(url)

#%%
lines = r.text.split('\n') # pull the text from the response object and split based on new lines
data = [l.split('\t') for l in lines] # split each line into its components based on tabs
# convert to dataframe using the first row as the column names; drop empty, final row
df = pd.DataFrame(data[1:-1], columns = data[0])
# dataframe with the preferred names of the two proteins and the score of the interaction
interactions = df[['preferredName_A', 'preferredName_B', 'score']]

#%%
G = nx.Graph ( name='Protein Interaction Graph' )
interactions = np.array ( interactions )  # convert to array for clarity
for i in range ( len ( interactions ) ):
    interaction = interactions[ i ]
    a = interaction[ 0 ]  # protein a node
    b = interaction[ 1 ]  # protein b node
    w = int ( float ( interaction[ 2 ] ) * 100 )  # score as weighted edge

    # To include all the weighted connections, uncomment the following line
    # G.add_weighted_edges_from([(a,b,w)])

    # To only keep high scoring edges, use the following lines
    if w > 80:  # only keep high scoring edges
        G.add_weighted_edges_from ( [ (a , b , w) ] )
#%% simple plot
c = CircosPlot(G,node_labels=True)
c.draw()
plt.show()

#%% drax graph using "class" and "weight" to display info

# function to rescale list of values to range [newmin,newmax]
def rescale(l,newmin,newmax,rnd=False):
    arr = list(l)
    return [round((x-min(arr))/(max(arr)-min(arr))*(newmax-newmin)+newmin,2) for x in arr]

nodelist = [n for n in G.nodes]
ws = rescale([float(G[u][v]['weight']) for u,v in G.edges],1,10)
# alternative method below
# ws = rescale([float(G[u][v]['weight'])**70 for u,v in G.edges],1,50)
edgelist = [(str(u),str(v),{"weight":ws.pop(0)}) for u,v in G.edges]

# create new graph using nodelist and edgelist
g = nx.Graph(name='Protein Interaction Graph')
g.add_nodes_from(nodelist)
g.add_edges_from(edgelist)
# go through nodes in graph G and store their degree as "class" in graph g
for v in G:
    g.nodes[v]["class"] = G.degree(v)


c = CircosPlot(graph=g,figsize=(13, 13),node_grouping="class", node_color="class",
               edge_width="weight",node_labels=True)
c.draw()
plt.show()