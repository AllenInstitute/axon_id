import pandas as pd
import numpy as np
from itertools import combinations
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

def extract_features(mshwks): 
    
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
    final_df = pd.DataFrame(data = None, columns = ['root_id', 'supervoxel_id', 'soma_pt', 'segment', 'ctr_pt',  
                                            'length', 'pre', 'n_pre', 'post', 'n_post', 'total_syn', 'density',
                                            'soma_dist']) 

    soma_df = client.materialize.query_table('nucleus_neuron_svm')


    for msh in mshwks:
        seg_id = msh.seg_id
        # update the seg_id if necessary 
        if client.chunkedgraph.is_latest_roots([int(msh.seg_id)]):
            updated_seg_id = msh.seg_id
        else:
            updated_seg_id = neuron_io.get_root_id_from_point(msh.skeleton.vertices[msh.skeleton.root])
            print('updated root is ' + str(updated_seg_id))

        body_df =pd.DataFrame(data = None, columns = ['root_id', 'supervoxel_id', 'soma_pt', 'segment', 'ctr_pt', 
                                              'length', 'pre', 'n_pre', 'post', 'n_post', 'total_syn', 'density',
                                              'soma_dist']) 
        segs = list(map(list,msh.skeleton.segments))

        body_df.loc[:,'segment'] = segs

        for i in range(len(body_df)):

            seg = body_df.loc[i,'segment']

            body_df.loc[i, 'root_id'] = seg_id
            
            body_df.loc[i, 'supervoxel_id'] = neuron_io.root_to_supervoxel([updated_seg_id], soma_df)[0]

            body_df.loc[i, 'soma_pt'] = msh.skeleton.root

            body_df.loc[i, 'ctr_pt'] = seg[len(seg)//2]

            body_df.loc[i, 'length'] = len(seg)
        
            body_df.at[i, 'pre'] = set(seg) & set(msh.skeleton.mesh_to_skel_map[msh.anno.pre_syn.mesh_index])
            body_df.loc[i, 'n_pre'] = len(body_df.loc[i, 'pre'])
            
            body_df.at[i, 'post'] = set(seg) & set(msh.skeleton.mesh_to_skel_map[msh.anno.post_syn.mesh_index])
            body_df.loc[i, 'n_post'] = len(body_df.loc[i, 'post'])
            
            body_df.loc[i, 'total_syn'] = body_df.loc[i, 'n_pre'] + body_df.loc[i, 'n_post']
            body_df.loc[i, 'density'] = body_df.loc[i, 'total_syn']/body_df.loc[i, 'length']
            
            body_df.loc[i, 'soma_dist'] = len(msh.skeleton.path_between(int(msh.skeleton.root), msh.skeleton.segments[i][-1]))

        final_df = pd.concat([final_df, body_df.replace(np.nan, '-')])
        
    final_df.reset_index(inplace = True, drop = True)
        
    return final_df
        
            








# returns the classification of the branch that each for each skel index belongs to 
# and assigns that skel index the corrosponding color for that classification. 


      
def classify_axon_dendrite(msh, model):
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
    
    #creating the df. type = np.float to allow user to run sns.pairplot later. 
    classified_segments_df = pd.DataFrame(data = None, columns = ['meshwork', 'root_id', 'soma_pt', 'segment', 'ctr_pt', 'length', 
                                              'pre', 'n_pre', 'post', 'n_post', 'total_syn', 'density',
                                              'soma_dist', 'classification'], dtype = np.float64) 
                                              # df from all meshworks in meshworks list
                                              # added density, 
    

    classified_segments_df['segment'] = msh.skeleton.segments

    # need to do this because thesthe cells in this columns will contain a set
    classified_segments_df['pre'] = classified_segments_df['pre'].astype('object')
    classified_segments_df['post'] = classified_segments_df['post'].astype('object')

    for i in range(len(classified_segments_df)):

            classified_segments_df.loc[i, 'meshwork'] = msh

            if hasattr(msh.skeleton, 'seg_id'):
                classified_segments_df.loc[i, 'root_id'] = msh.skeleton.seg_id

            classified_segments_df.loc[i, 'soma_pt'] = int(msh.skeleton.root)
            
            seg = classified_segments_df.loc[i, 'segment']
            
            classified_segments_df.loc[i, 'ctr_pt'] = seg[len(seg)//2]
            
            classified_segments_df.loc[i, 'length'] = len(seg)
            
            classified_segments_df.at[i, 'pre'] = set(seg) & set(msh.skeleton.mesh_to_skel_map[msh.anno.pre_syn.mesh_index])
            classified_segments_df.loc[i, 'n_pre'] = len(classified_segments_df.loc[i, 'pre'])
            
            classified_segments_df.at[i, 'post'] = set(seg) & set(msh.skeleton.mesh_to_skel_map[msh.anno.post_syn.mesh_index])
            classified_segments_df.loc[i, 'n_post'] = len(classified_segments_df.loc[i, 'post'])
            
            classified_segments_df.loc[i, 'total_syn'] = classified_segments_df.loc[i, 'n_pre'] + classified_segments_df.loc[i, 'n_post']
            classified_segments_df.loc[i, 'density'] = classified_segments_df.loc[i, 'total_syn']/classified_segments_df.loc[i, 'length']
            
            classified_segments_df.loc[i, 'soma_dist'] = len(msh.path_between(int(msh.skeleton.seg_id), msh.skeleton.segments[i][-1], return_as_skel = True))

            # add final classificaiton to the segment



    classified_segments_df['classification'] = model.predict(classified_segments_df.drop(['classification', 'soma_pt', 'meshwork', 'root_id', 'segment', 'soma_pt', 'pre', 'post'], axis = 1))

        
    classified_segments_df.reset_index(inplace = True, drop = True)
        
    return classified_segments_df





def segment_graph(skel, df):
    
    '''
    
    creates a graph of each segment-neighbor pair, where each segment is represented by the segment centerpoint 
    
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
    seg_id_mw_dict = neuron_io.load_mws_from_cloud(cf_skel_source_path, asdict = True, update_roots = True)
    
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

    mws = neuron_io.load_mws_from_cloud(mws_cf_source_path)

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

        neuron_io.write_meshwork_h5_to_cf(mw, cloud_path)

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

    neuron_io.write_meshwork_h5_to_cf(mw, cloud_path)

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
    X1 = features_df.drop(['supervoxel_id', 'soma_pt', 'root_id', 'segment', 'pre', 'post'], axis = 1)
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
    X2 = classification_df_rf1_neighbors.drop(['classification', 'supervoxel_id', 'soma_pt', 'root_id', 'segment', 'pre', 'post'], axis = 1)
    predicted_segment_classifications_rf2 = rf2.predict(X2)
    # save results to memory
    classification_df_rf2 = classification_df_rf1_neighbors.copy()
    classification_df_rf2['predicted classification rf2'] = predicted_segment_classifications_rf2
    return classification_df_rf2

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