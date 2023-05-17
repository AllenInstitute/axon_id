import pandas as pd
import numpy as np
from itertools import combinations, permutations
import joblib
import os
import io

import cloudvolume
from cloudfiles import CloudFiles
from caveclient import CAVEclient
from meshparty import meshwork
from taskqueue import queueable
from meshparty import meshwork

from axon_id import neuron_io, models

client = CAVEclient('minnie65_phase3_v1')
cv = cloudvolume.CloudVolume(client.info.segmentation_source(), progress = False, use_https = True, parallel=24)

# TO DO : READ MODELS HERE so they dont have to be loaded every time


            

def extract_features(mw, soma_seg_len = 100): 
    
    '''
    takes in a skeleton meshwork and extracts the features for each segment into a df 



    Parameters
    ----------
    mw : meshparty.meshwork.meshwork.Meshwork
        meshwork of the bodies to have their features extracted
  
    Returns
    -------
    final_df : pd.DataFrame()
        data frame that contains all the segments in all of the skels and their ectracted features. 
    
    
    '''
    feat_df = pd.DataFrame(data = None, columns = ['root_id', 'soma_id', 'soma_pt', 'segment', 'ctr_pt',  
                                            'length', 'pre', 'n_pre', 'pre_size', 'post', 'n_post', 'post_size', 
                                            'total_syn', 'density', 'soma_dist', 'radius', 'endpoint']) 


    
    segs = mw.skeleton.segments
    

    feat_df.loc[:,'segment'] = segs

    seg_df = mw.anno.segment_properties.df.set_index('mesh_ind_filt')

    endpts = mw.skeleton.end_points

    for i in range(len(feat_df)):

        seg = mw.skeleton.segments[i]

        feat_df.loc[i, 'root_id'] = mw.seg_id
        
        feat_df.loc[i, 'soma_id'] = mw.anno.soma_row['id'][0]

        feat_df.loc[i, 'soma_pt'] = mw.skeleton.root

        feat_df.loc[i, 'ctr_pt'] = seg[len(seg)//2]

        # for length, I want to include the parent node unless seg is just soma 
        
        if sum(mw.skeleton.root == seg) == 1:
            feat_df.loc[i, 'length'] = soma_seg_len
        else:
            len_seg = list(seg)
            len_seg.append(int(mw.skeleton.parent_nodes(seg[-1])))
            len_seg = mw.skeleton.SkeletonIndex(len_seg)
            feat_df.loc[i, 'length'] = mw.skeleton.path_length(len_seg)

        # pull out mesh indices of seg 

        msh_seg = [mw.SkeletonIndex(s).to_mesh_index for s in seg]
        # flatten
        msh_seg = [item for sublist in msh_seg for item in sublist]

        presyn_df = mw.anno.pre_syn.df.set_index('pre_pt_mesh_ind')
        presyn_df = presyn_df[presyn_df.index.isin(msh_seg)]
        pre_msh_inds = mw.anno.pre_syn.mesh_index
        feat_df.at[i, 'pre'] = np.intersect1d(seg, mw.skeleton.mesh_to_skel_map[pre_msh_inds])
        feat_df.loc[i, 'n_pre'] = len(presyn_df)
        feat_df.at[i, 'pre_size'] = presyn_df['size'].sum()

        
        postsyn_df = mw.anno.post_syn.df.set_index('post_pt_mesh_ind')
        postsyn_df = postsyn_df[postsyn_df.index.isin(msh_seg)]
        post_msh_inds = mw.anno.post_syn.mesh_index
        feat_df.at[i, 'post'] = np.intersect1d(seg, mw.skeleton.mesh_to_skel_map[post_msh_inds])
        feat_df.loc[i, 'n_post'] = len(postsyn_df)
        feat_df.at[i, 'post_size'] = postsyn_df['size'].sum()

        feat_df.loc[i, 'total_syn'] = feat_df.loc[i, 'n_pre'] + feat_df.loc[i, 'n_post']
        feat_df.loc[i, 'density'] = feat_df.loc[i, 'total_syn']/feat_df.loc[i, 'length']
        feat_df.loc[i, 'soma_dist'] = mw.skeleton.path_length(mw.skeleton.path_between(int(mw.skeleton.root), seg[-1]))

        feat_df.loc[i, 'radius'] = np.mean(seg_df.loc[seg_df.index.isin(msh_seg)]['r_eff'])
        
        feat_df.loc[i, 'endpoint'] = bool([x for x in seg if x in endpts])

    feat_df = feat_df.replace(np.nan, '-')
        
    return feat_df

def extract_features_multiple(mws, soma_seg_len = 100): 
    
    '''
    takes in a list of skeleton meshworks and extracts the features for each segmentinto a df 



    Parameters
    ----------
    skels : list of meshparty.meshwork.meshwork.Meshworks
        meshwork of the bodies to have their features extracted
  
    Returns
    -------
    final_df : pd.DataFrame()
        data frame that contains all the segments in all of the skels and their ectracted features. 
    
    
    '''
    final_df = pd.DataFrame(data = None, columns = ['root_id', 'soma_id', 'soma_pt', 'segment', 'ctr_pt',  
                                            'length', 'pre', 'n_pre', 'pre_size', 'post', 'n_post', 'post_size', 
                                            'total_syn', 'density', 'soma_dist', 'radius', 'endpoint']) 



    for mw in mws:
        body_df = extract_features(mw, soma_seg_len = soma_seg_len)

        final_df = pd.concat([final_df, body_df.replace(np.nan, '-')])
        
    final_df.reset_index(inplace = True, drop = True)
        
    return final_df
        




# returns the classification of the branch that each for each skel index belongs to 
# and assigns that skel index the corrosponding color for that classification. 


      
def classify_axon_dendrite(mw, model, model_columns = ['length', 'n_pre', 'pre_size','n_post', 'post_size', 
                                            'total_syn', 'density', 'soma_dist', 'radius', 'endpoint'],
                                            soma_seg_len = 100):
    '''
    takes in a meshwork file and classification model and classifies each segment in the body



    Parameters
    ----------
    msh : meshparty.meshwork.meshwork.Meshwork
        meshwork of the body to be classified
    model : 
        the model with which to classify axons and dendrites
  
    Returns
    -------
    classified_segments_df : pd.DataFrame
        dataframe which contains each segment and the classificaiton
    
    
    '''
    
    feats_class_df = extract_features(mw, soma_seg_len = soma_seg_len)
    
    feats_class_df['classification'] = model.predict(feats_class_df[model_columns], axis = 1)

    feats_class_df.reset_index(inplace = True, drop = True)
        
    return feats_class_df


def downstream_segment_graph(sk):
    '''
    creates a df with column:
    'a' - the center point of the first segment of an edge.
    'class a' - the class of the first segment.
    'b' - the center point of the second segment of an edge.
    'class b' - the class of the second segment.

    '''

    seg_connectivity_df = pd.DataFrame(columns = ['upstream_seg', 'downstream_seg'])

    for skidx in sk.branch_points:

        segment_idx = sk.segment_map[skidx]
        segment = sk.segments[segment_idx]

        child_nodes = sk.child_nodes(skidx)
        for node in child_nodes:
            child_seg_idx = sk.segment_map[node]
            child_seg = sk.segments[child_seg_idx]
            new_row = [segment, child_seg]
            seg_connectivity_df.loc[len(seg_connectivity_df)] = new_row
    
    return seg_connectivity_df

def sibling_segment_graph(sk, col_names = ['upstream_seg', 'downstream_seg']):
    '''
    creates a df tracking when a -> [b, c, d], b c and d are 'siblings'
    '''

    seg_connectivity_df = pd.DataFrame(columns = col_names)


    for skidx in sk.branch_points:

        child_nodes = sk.child_nodes(skidx)
        child_segs = sk.segment_map[child_nodes]

        # now get combos of the child segs
        lil_connectivity_df = pd.DataFrame(list(permutations(child_segs, 2)), 
                                            columns = col_names)
        seg_connectivity_df = seg_connectivity_df.append(lil_connectivity_df)
    return seg_connectivity_df



def segment_graph(skel, df):
    
    '''
    
    creates a graph of each segment-neighbor pair, where each segment is represented by the segment centerpoint 
    df must have 'ctr_pt' and 'classification
    '''
    
    
    seg_connectivity_df = pd.DataFrame(columns = ['a', 'class a', 'b', 'class b'])
    running_combo_df = pd.DataFrame(columns = ['a', 'class a', 'b', 'class b'])
    
    
    for skidx in skel.branch_points:
        
        segment_idx = skel.segment_map[skidx]
        segment = skel.segments[segment_idx]
        ctrpt = segment[len(segment)//2]
        segment_class = df.loc[df['ctr_pt'] == ctrpt,'classification'].iloc[0]
        
        child_nodes = skel.child_nodes(skidx)
        
        for node in child_nodes:
            child_seg_idx = skel.segment_map[node]
            child_seg = skel.segments[child_seg_idx]
            child_ctrpt = child_seg[len(child_seg)//2]
            child_class = df.loc[df['ctr_pt'] == child_ctrpt,'classification'].iloc[0]
            
            new_row = [ctrpt, segment_class, child_ctrpt, child_class]
            seg_connectivity_df.loc[len(seg_connectivity_df)] = new_row
            
    
    # what about when a -> b & c & d. need to indicate that b and c and d are all connected to each other. 
    # so say get all the unordered combinations of b, c, and d. add each as a row. 
        
    
    for skelidx in list(seg_connectivity_df['a'].unique()):
        
        # list all the downstream segments from each branch point
        connected_list = list(seg_connectivity_df.loc[seg_connectivity_df['a']==skelidx, 'b'])
        
        # find the combinations of all of those downstream segments, as they are all also connected to each other.
        combos = list(combinations(connected_list, 2))
        
        # create a df to store those combos in 
        combodf = pd.DataFrame(columns = ['a', 'class a', 'b', 'class b'])
        
        # store these combos as a and b 
        combodf['a'], combodf['b'] = zip(*combos)
        
        # for each row in the combodf, add the classes for a and b 
        
        
        
        for i in range(len(combodf)):
            combodf.loc[i, 'class a'] = seg_connectivity_df.loc[seg_connectivity_df['b']==combodf.loc[i, 'a'],['class b']].iloc[0,0]
            combodf.loc[i, 'class b'] = seg_connectivity_df.loc[seg_connectivity_df['b']==combodf.loc[i, 'b'],['class b']].iloc[0,0]
            
            # add the combo df to the end of the connectivity df 
            running_combo_df = running_combo_df.append(combodf)
            #seg_connectivity_df.drop_duplicates(inplace = True)
            #return(seg_connectivity_df)
    
    # add the running combo df (which contains all combos) to the seg connectivity df
    seg_connectivity_df = seg_connectivity_df.append(running_combo_df)
    
    # we are making this an undirected graph, so add the b columns to a and their corresponding a columns to b
    reversed_seg_connectivity_df = seg_connectivity_df.copy()
    reversed_seg_connectivity_df.loc[:, ['a', 'class a', 'b', 'class b']] = seg_connectivity_df.loc[:, ['b', 'class b', 'a', 'class a']].values
    
    # add the two graphs together 
    final_seg_connectivity_df = pd.concat([seg_connectivity_df, reversed_seg_connectivity_df])
    
    final_seg_connectivity_df.drop_duplicates(inplace = True)
    final_seg_connectivity_df.reset_index(inplace = True, drop = True)
    final_seg_connectivity_df.sort_values(by=['a'], inplace = True)
    
    return final_seg_connectivity_df 
        
def neighboring_segments(skel, df):
    
    '''
    
    takes in a mesh and its corresponding segment features and classification df and adds a column with the 
    classification of neighboring segments to each segment. 
    
    '''
    df.reset_index(inplace = True, drop = True)
    
    df.loc[:,'axon neighbor count'] = 0
    df.loc[:,'dendrite neighbor count'] = 0
    
    # get the graph of which segments are connected to other segments
    segment_connectivity_graph = segment_graph(skel, df)
    
    for i in range(len(df)):
        
        ctrpt = df.loc[i, 'ctr_pt']
        
        # use segment connectivity graph to find the number of connected segments of each class for each segment in this skeleton
        df.loc[i, 'dendrite neighbor count'] = len(segment_connectivity_graph.loc[(segment_connectivity_graph.loc[:,'a'] == ctrpt) & (segment_connectivity_graph.loc[:, 'class b'] == 0.0)])
        df.loc[i, 'axon neighbor count'] = len(segment_connectivity_graph.loc[(segment_connectivity_graph.loc[:,'a'] == ctrpt) & (segment_connectivity_graph.loc[:, 'class b'] == 1.0)])
        
        
    return df



def neighboring_segments_multiple_bodies(df, cf_skel_source_path):
    
    '''
    df - dataframe, features df with round 1 of classification done
    cf_skel_source_path - string, name of folder inside of allen-minnie-phase3/minniephase3-emily-pcg-skeletons

    takes in features df after 1st round of classification. 
    creates two columns counting the number of neighboring axons and 
    neighboring dendrites for each segment 

    pulls skeletons from a specified cloudfiles folder in gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons
    
    '''
    
    #final df that will contain neighboring segment information
    final_df = pd.DataFrame(columns = df.columns)
    # mapping of seg id to mw file pulled from speficied cloudfile folder 
    seg_id_mw_dict = neuron_io.load_mws_from_folder(cf_skel_source_path, asdict = True, update_roots = True)
    
    #iterate through each unique body in the features df
    for sup_id in df['supervoxel_id'].unique():

        updated_root_id = client.chunkedgraph.get_root_id(sup_id)

        body_df = df.loc[df['supervoxel_id'] == sup_id]

        # find the corresponding skeleton to this root_id
        skel = seg_id_mw_dict[updated_root_id].skeleton

        # get the graph of each segment-neighboring segment pair 
        body_df_with_neighbors = neighboring_segments(skel, body_df)
        
        final_df = pd.concat([final_df, body_df_with_neighbors])
        
    final_df.reset_index(inplace = True, drop = True)
    return final_df

def _skel_dendrite_map(df):

    '''
    takes in features df, returns a map of all skeleton indices and boolean of weather or not that
    index is on a dendrite classified segment 

    '''
    
    map_df = pd.DataFrame(columns = ['skeleton index', 'tf dendrite'])
    tf_dendrite_dict = {0:True, 1:False}
    
    for i in range(len(df)):
        segment = df.loc[i, 'segment']
        tf_dendrite = tf_dendrite_dict[df.loc[i, 'predicted classification rf2']]
        
        for node in segment:
            map_df.loc[int(node), 'skeleton index'] = int(node)
            map_df.loc[int(node), 'tf dendrite'] = tf_dendrite
    
    map_df.sort_values(['skeleton index'], inplace=True)
    map_df.reset_index(inplace=True, drop=True)
    
    return map_df



# Put it all together
def remove_axons_multiple_bodies(mws_cf_source_path, skels_cf_destination_path, m1, m2, verbose = True):
    '''

    Takes in a skeleton object, returns a skeleton object without the identified axons. 
    
    Parameters
    ----------
    mshwks : list of meshparty.skeleton.Skeleton objects
        skeleton objects to have their axons removed 
    mshwks_cf_path : string
        path to folder to save resulting skeletons (cloudfiles format)
    m1 : string
        path to m1
    ....

    Returns
    -------
    skel : list of meshparty.skeleton.Skeleton objects
        origional objects with all axon segments removed

    '''

    mws = neuron_io.load_mws_from_folder(mws_cf_source_path)

    print('extracting features...')
    features_df = extract_features(mws)

    # set up cloud path 
    cloud_root = os.environ.get('SAVE_LOCATION', 'gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons')
    cloud_path = os.path.join(cloud_root, skels_cf_destination_path)

    # import the ML models
    # random forest model 1: classify axon/dendrite round 1
    rf1 = joblib.load(m1)
    # random forest model 2: classify axon dendrite with round 1 neighboring segment classification as feature
    rf2 = joblib.load(m2)

    # apply rf1 to features df
    print('applying model 1...')
    X1 = features_df.drop(['supervoxel_id', 'soma_pt', 'root_id', 'segment', 'pre', 'post'], axis = 1)
    predicted_segment_classifications_rf1 = rf1.predict(X1)
    # save results to memory
    classification_df_rf1 = features_df.copy()
    classification_df_rf1['classification'] = predicted_segment_classifications_rf1

    # add columns indicating neighboring segment classification for each segment
    print('adding neighboring segment classification features...')
    classification_df_rf1_neighbors = neighboring_segments_multiple_bodies(classification_df_rf1, cf_skel_source_path = mws_cf_source_path)
    # priblem - I need the skeleton itself here... mayeb I can pull from google cloud

    # apply rf2 to features df
    print('applying model 2...')
    X2 = classification_df_rf1_neighbors.drop(['classification', 'supervoxel_id', 'soma_pt', 'root_id', 'segment', 'pre', 'post'], axis = 1)
    predicted_segment_classifications_rf2 = rf2.predict(X2)
    # save results to memory
    classification_df_rf2 = classification_df_rf1_neighbors.copy()
    classification_df_rf2['predicted classification rf2'] = predicted_segment_classifications_rf2

    # recreate the meshwork files to include only their dendrite classified segments
    # I MIGHT NEED TO FIX THE SEGMENT COLUMN TO BE A LIST HERE
    # mask the skel to only include nodes that are classified to belong to a dendrite segment 
    for mw in mws:
        print('skeletonizing ' + str(mw.seg_id) + "...")
        # pull out the classification df for this meshwork  
        print('pulling body df from features df') 
        skel_df = classification_df_rf2[features_df['root_id'] == mw.seg_id]
        print(len(skel_df))
        print(skel_df)
        skel_df.reset_index(inplace=True, drop = True)
        #if len(skel_df) == 0:
            #raise ValueError(f'This segment id {mw.seg_id} does not appear in features_df')
        # create and apply mask
        # create mesh mask 
        print('creating mesh mask') 
        dendrite_mask = _skel_dendrite_map(skel_df)
        my_verts = mw.SkeletonIndex(dendrite_mask[dendrite_mask['tf dendrite'] == True]['skeleton index'])
        print('applying mesh mask') 
        mw.apply_mask(my_verts.to_mesh_mask)

        print('saving to cloud') 

        neuron_io.write_meshwork_h5_to_folder(mw, cloud_path)

        print(str(mw.seg_id) + " skeletonized and saved to cloud")

            
            
def remove_axons(mw, mws_cf_destination_path, m1, m2):

    '''

    Takes in a meshwork object, uses two ml models to identify which segments are dendrites and which segments are axons
    saves the resulting masked meshwork (which only contains the dendrites) to the speficied cloudfiles destination path

    Parameters
    ----------
    mw : meshparty.skeleton.Skeleton object
        skeleton object to have its axons removed 
    mshwks_cf_path : string
        path to folder to save resulting masked meshworks (cloudfiles format)
    m1 : string
        path to the first ML model
    m2 : string
        path to the second ML model

    Returns
    -------
    skel : meshparty.skeleton.Skeleton object
        origional object with all axon segments removed, saved to cloud

    '''

    # set up cloud path 
    cloud_root = os.environ.get('SAVE_LOCATION', 'gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons')
    cloud_path = os.path.join(cloud_root, mws_cf_destination_path)

    classification_df_rf2 = make_classification_df(mw, m1, m2)

    # recreate the meshwork files to include only their dendrite classified segments
    print('skeletonizing ' + str(mw.seg_id) + "...")

    # create and apply mask
    # create mesh mask 
    print('creating mesh mask') 
    dendrite_mask = _skel_dendrite_map(classification_df_rf2)
    my_verts = mw.SkeletonIndex(dendrite_mask[dendrite_mask['tf dendrite'] == True]['skeleton index'])
    print('applying mesh mask') 
    print(dendrite_mask)
    mw.apply_mask(my_verts.to_mesh_mask)

    print('saving to cloud') 

    neuron_io.write_meshwork_h5_to_folder(mw, cloud_path)

    print(str(mw.seg_id) + " skeletonized and saved to cloud")

def make_classification_df(mw, m1, m2):
    '''creates the classficiation df after 2 rounds of axon identification with 2 models'''
    # import the ML models
    # random forest model 1: classify axon/dendrite round 1
    rf1 = joblib.load(m1)
    # random forest model 2: classify axon dendrite with round 1 neighboring segment classification as feature
    rf2 = joblib.load(m2)
   # extract features
    print('extracting features for body ' + str(mw.seg_id) + '...')
    features_df = extract_features([mw])

    # apply rf1 to features df
    print('applying model 1...')
    X1 = features_df.drop(['soma_pt', 'root_id', 'soma_id', 'segment', 'pre', 'post', 'pre_size', 'post_size', 'radius'], axis = 1)
    predicted_segment_classifications_rf1 = rf1.predict(X1)
    # save results to memory
    classification_df_rf1 = features_df.copy()
    classification_df_rf1['classification'] = predicted_segment_classifications_rf1

    # add columns indicating neighboring segment classification for each segment
    print('adding neighboring segment classification features...')
    classification_df_rf1_neighbors = neighboring_segments(mw.skeleton, classification_df_rf1)
    # priblem - I need the skeleton itself here... mayeb I can pull from google cloud

    # apply rf2 to features df
    print('applying model 2...')
    X2 = classification_df_rf1_neighbors.drop(['classification', 'soma_pt', 'root_id', 'soma_id', 'segment', 'pre', 'post', 'pre_size', 'post_size', 'radius'], axis = 1)
    predicted_segment_classifications_rf2 = rf2.predict(X2)
    # save results to memory
    classification_df_rf2 = classification_df_rf1_neighbors.copy()
    classification_df_rf2['predicted classification rf2'] = predicted_segment_classifications_rf2
    return classification_df_rf2

def apply_first_class(mw, m1):
    rf1 = joblib.load(m1)
    features_df = extract_features([mw])
    X1 = features_df.drop(['soma_pt', 'root_id', 'soma_id', 'segment', 'pre', 'post', 'pre_size', 'post_size', 'radius'], axis = 1)
    predicted_segment_classifications_rf1 = rf1.predict(X1)
    classification_df_rf1 = features_df.copy()
    classification_df_rf1['classification'] = predicted_segment_classifications_rf1
    return classification_df_rf1

@queueable
def remove_axons_tq(source_cloud_folder, filename, destination_cloud_file, m1, m2):
    
    '''
    Parameters
    ----------
    cf: cloudfiles.cloudfiles.CloudFiles
        the cloudfiles object that contains the folder in which 'filename'
        is stored 
        i.e. 
        cf = CloudFiles(axon_id.neuron_io.add_cloud_path('cf_source_folder'))
    filename : str
        the name of the file in cf to have its axons removed
    destination_cloud_file : str
        the folder in the cloud path specified in .env.docker
    m1 : str
        path to the first ml model to remove axons
    m2 : str
        path to the second ml model to remove axons
    
    '''
    cf = CloudFiles(neuron_io.add_cloud_path(source_cloud_folder))
    # cloadfiles removes the filename as bytes
    # read those bytes into a BytesIO obj
    with io.BytesIO(cf.get(filename)) as f:
        # define the starting point as the first element 
        f.seek(0)
        # load those bytes as mw obj
        mw = meshwork.load_meshwork(f)
        
        # remove axons from that obj
        models.remove_axons(mw, destination_cloud_file, m1, m2)