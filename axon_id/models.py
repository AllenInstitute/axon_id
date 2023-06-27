import pandas as pd
import numpy as np
import joblib


from axon_id import graph
            

def extract_features(mw, soma_seg_len = 10000): 
    
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
    
    # now add neighbor feats 
    upstream_df = graph.downstream_segment_graph(mw.skeleton, columns = ['upstream_seg', 'downstream_seg'],
                                sortcol = 'upstream_seg')
    feat_df['self/parent_rad'] = [graph.parent_rad(seg, feat_df, upstream_df) for seg in range(len(feat_df))]
        
    return feat_df






# def add_corresponding_columns(df_A, df_B, column_name, df_B_col_a = 'upstream_seg', 
#                     df_B_col_b = 'downstream_seg'):

#     '''
#     pulls column from df A onto df B from indices in df B columns a and b
    
#     '''
#     # Create a series mapping segment numbers to the specified column in df_A

#     column_mapping = df_A[column_name]

#     # Add a new column to df_B with the mapped values for both upstream and downstream segments
#     df_B['Upstream_'+column_name] = df_B['Upstream_Segment'].map(column_mapping)
#     df_B['Downstream_'+column_name] = df_B['Downstream_Segment'].map(column_mapping)

#     return df_B


# def add_neighbor_feats(feat_df, graph_df, myseg_colname = 'a', familyseg_colname = 'b', 
#                                     featurename = 'radius', agg_func = np.mean, 
#                                     new_col_name = 'upstream_radius'):
#     '''
#     this function will take a feature df, then add a column that aggregates
#     the data in graph df 
    
#     '''
#     # iterate through feat df and get info for that column 
#     feat_df[new_col_name] = np.nan
#     for i in range(len(feat_df)):
#         # get relevant partners to your seg 
#         seg_graph = graph_df[graph_df[myseg_colname] == i]

#         # agg that 
#         agg = agg_func(seg_graph[featurename])

#         # add it to the df 
#         feat_df.loc[i, new_col_name] = agg
    
#     return feat_df







        




# returns the classification of the branch that each for each skel index belongs to 
# and assigns that skel index the corrosponding color for that classification. 


      
def classify_axon_dendrite(mw, model_path, model_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
                                'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad'],
                                            soma_seg_len = 10000):
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
    
    model = joblib.load(model_path)

    feats_class_df['classification'] = model.predict(feats_class_df[model_columns])

    feats_class_df.reset_index(inplace = True, drop = True)
        
    return feats_class_df

def classify_axon_dendrite_2(mw, model1_path, model2_path, model2_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
       'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad','classification', 'axon neighbor count', 
       'dendrite neighbor count', 'upstream_class'], 
       model1_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
                                'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad'], soma_seg_len = 10000):
    
    '''
    takes mw and the path to model 1 and model 2, applies 2 rounds of classification
    
    '''

    m1_df = classify_axon_dendrite(mw, model1_path, model_columns = model1_columns,
                                            soma_seg_len = 10000)

    model2 = joblib.load(model2_path)

    # get neighboring features 
    m2_df = extract_features_2(mw, m1_df)

    m2_df['r2_classification'] = model2.predict(m2_df[model2_columns])
    m2_df.reset_index(inplace = True, drop = True)
    return m2_df  



    


# def sibling_segment_graph(sk, col_names = ['a', 'b'], 
#                             sortcol = 'a'):
#     '''
#     creates a df tracking when a -> [b, c, d], b c and d are 'siblings'
#     '''

#     seg_connectivity_df = pd.DataFrame(columns = col_names)


#     for skidx in sk.branch_points:

#         child_nodes = sk.child_nodes(skidx)
#         child_segs = sk.segment_map[child_nodes]

#         # now get combos of the child segs
#         lil_connectivity_df = pd.DataFrame(list(permutations(child_segs, 2)), 
#                                             columns = col_names)
#         seg_connectivity_df = seg_connectivity_df.append(lil_connectivity_df)

#     seg_connectivity_df.drop_duplicates(inplace = True)
#     seg_connectivity_df.reset_index(inplace = True, drop = True)
#     seg_connectivity_df.sort_values(by=[sortcol], inplace = True)
#     return seg_connectivity_df

# def create_seg_graph(feature_df, partner_seg_df, 
#                         seg_column = 'upstream_seg', 
#                         partner_column = 'downstream_seg',
#                         feat_colname = 'classification', 
#                         agglom = sum, newfeat_col = 'downstream_classification'):
#     '''
#     takes a df with the each segment and some measured feat or classification
#     for that segment, then creates a new column in the feature df that 
#     agglomerates that feature by matching the 
#     seg column is the column in the partner seg df that represents the segment in 
#     feature df 
    
#     '''

#     # go line by line in the feature df. 
#     # the index of this df is the segment index
#     feature_df[newfeat_col] = [np.nan]*len(feature_df)

#     for seg_ind in range(len(feature_df)):
#         # seg_ind is current segment index

#         # get partner seg inds where the 'seg column' equals seg_ind
#         seg_partners = partner_seg_df[partner_seg_df[seg_column] == seg_ind][partner_column]
#         # get the agglom of the feature
#         new_ft = feature_df.iloc[seg_partners][feat_colname].agglom

#         # add this new feat to the row 
#         feature_df.loc[seg_ind, newfeat_col] = new_ft

#     return feature_df







def extract_features_2(mw, m1_class_df):
    '''
    adds neigboring classes for round 2 classification 
    '''

    # add the number of neighbors that are axon or dendrite 
    r2_df = graph.neighboring_segments(mw.skeleton, m1_class_df)

    # add if parent segment is axon or dendrite 
    downstream_seg_graph = graph.downstream_segment_graph(mw.skeleton)
    r2_df['upstream_class'] = [graph.parent_class(seg, m1_class_df, downstream_seg_graph) for seg in r2_df.index]

    return r2_df

    




# def neighboring_segments_multiple_bodies(df, cf_skel_source_path, client):
    
#     '''
#     df - dataframe, features df with round 1 of classification done
#     cf_skel_source_path - string, name of folder inside of allen-minnie-phase3/minniephase3-emily-pcg-skeletons

#     takes in features df after 1st round of classification. 
#     creates two columns counting the number of neighboring axons and 
#     neighboring dendrites for each segment 

#     pulls skeletons from a specified cloudfiles folder in gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons
    
#     '''
    
#     #final df that will contain neighboring segment information
#     final_df = pd.DataFrame(columns = df.columns)
#     # mapping of seg id to mw file pulled from speficied cloudfile folder 
#     seg_id_mw_dict = neuron_io.load_mws_from_folder(cf_skel_source_path, asdict = True, update_roots = True)
    
#     #iterate through each unique body in the features df
#     for sup_id in df['supervoxel_id'].unique():

#         updated_root_id = client.chunkedgraph.get_root_id(sup_id)

#         body_df = df.loc[df['supervoxel_id'] == sup_id]

#         # find the corresponding skeleton to this root_id
#         skel = seg_id_mw_dict[updated_root_id].skeleton

#         # get the graph of each segment-neighboring segment pair 
#         body_df_with_neighbors = neighboring_segments(skel, body_df)
        
#         final_df = pd.concat([final_df, body_df_with_neighbors])
        
#     final_df.reset_index(inplace = True, drop = True)
#     return final_df





# Put it all together

# function to take a mw, extract all features, then mask out the axons 

def mask_out_axons(mw, m1, m2, model2_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
       'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad','classification', 'axon neighbor count', 
       'dendrite neighbor count', 'upstream_class'], 
       model1_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
        'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad'], 
        soma_seg_len = 10000):

    # create m1 and m2 classification df 
    class_df = classify_axon_dendrite_2(mw, m1, m2, model2_columns = model2_columns, model1_columns = model1_columns) 

    # now get a map 
    skel_mask_df = graph.skel_dendrite_map(class_df, mw)
    skel_verts = mw.SkeletonIndex(skel_mask_df[skel_mask_df['tf dendrite']]['skeleton index'])
    # apply mask to skeleton
    mw.apply_mask(skel_verts.to_mesh_mask)
    return mw

# def remove_axons_multiple_bodies(mws_cf_source_path, skels_cf_destination_path, m1, m2, verbose = True):
#     '''

#     Takes in a skeleton object, returns a skeleton object without the identified axons. 
    
#     Parameters
#     ----------
#     mshwks : list of meshparty.skeleton.Skeleton objects
#         skeleton objects to have their axons removed 
#     mshwks_cf_path : string
#         path to folder to save resulting skeletons (cloudfiles format)
#     m1 : string
#         path to m1
#     ....

#     Returns
#     -------
#     skel : list of meshparty.skeleton.Skeleton objects
#         origional objects with all axon segments removed

#     '''

#     mws = neuron_io.load_mws_from_folder(mws_cf_source_path)

#     print('extracting features...')
#     features_df = extract_features(mws)

#     # set up cloud path 
#     cloud_root = os.environ.get('SAVE_LOCATION', 'gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons')
#     cloud_path = os.path.join(cloud_root, skels_cf_destination_path)

#     # import the ML models
#     # random forest model 1: classify axon/dendrite round 1
#     rf1 = joblib.load(m1)
#     # random forest model 2: classify axon dendrite with round 1 neighboring segment classification as feature
#     rf2 = joblib.load(m2)

#     # apply rf1 to features df
#     print('applying model 1...')
#     X1 = features_df.drop(['supervoxel_id', 'soma_pt', 'root_id', 'segment', 'pre', 'post'], axis = 1)
#     predicted_segment_classifications_rf1 = rf1.predict(X1)
#     # save results to memory
#     classification_df_rf1 = features_df.copy()
#     classification_df_rf1['classification'] = predicted_segment_classifications_rf1

#     # add columns indicating neighboring segment classification for each segment
#     print('adding neighboring segment classification features...')
#     classification_df_rf1_neighbors = neighboring_segments_multiple_bodies(classification_df_rf1, cf_skel_source_path = mws_cf_source_path)
#     # priblem - I need the skeleton itself here... mayeb I can pull from google cloud

#     # apply rf2 to features df
#     print('applying model 2...')
#     X2 = classification_df_rf1_neighbors.drop(['classification', 'supervoxel_id', 'soma_pt', 'root_id', 'segment', 'pre', 'post'], axis = 1)
#     predicted_segment_classifications_rf2 = rf2.predict(X2)
#     # save results to memory
#     classification_df_rf2 = classification_df_rf1_neighbors.copy()
#     classification_df_rf2['predicted classification rf2'] = predicted_segment_classifications_rf2

#     # recreate the meshwork files to include only their dendrite classified segments
#     # I MIGHT NEED TO FIX THE SEGMENT COLUMN TO BE A LIST HERE
#     # mask the skel to only include nodes that are classified to belong to a dendrite segment 
#     for mw in mws:
#         print('skeletonizing ' + str(mw.seg_id) + "...")
#         # pull out the classification df for this meshwork  
#         print('pulling body df from features df') 
#         skel_df = classification_df_rf2[features_df['root_id'] == mw.seg_id]
#         print(len(skel_df))
#         print(skel_df)
#         skel_df.reset_index(inplace=True, drop = True)
#         #if len(skel_df) == 0:
#             #raise ValueError(f'This segment id {mw.seg_id} does not appear in features_df')
#         # create and apply mask
#         # create mesh mask 
#         print('creating mesh mask') 
#         dendrite_mask = _skel_dendrite_map(skel_df)
#         my_verts = mw.SkeletonIndex(dendrite_mask[dendrite_mask['tf dendrite'] == True]['skeleton index'])
#         print('applying mesh mask') 
#         mw.apply_mask(my_verts.to_mesh_mask)

#         print('saving to cloud') 

#         neuron_io.write_meshwork_h5_to_folder(mw, cloud_path)

#         print(str(mw.seg_id) + " skeletonized and saved to cloud")

            
            
# def remove_axons(mw, mws_cf_destination_path, m1, m2):

#     '''

#     Takes in a meshwork object, uses two ml models to identify which segments are dendrites and which segments are axons
#     saves the resulting masked meshwork (which only contains the dendrites) to the speficied cloudfiles destination path

#     Parameters
#     ----------
#     mw : meshparty.skeleton.Skeleton object
#         skeleton object to have its axons removed 
#     mshwks_cf_path : string
#         path to folder to save resulting masked meshworks (cloudfiles format)
#     m1 : string
#         path to the first ML model
#     m2 : string
#         path to the second ML model

#     Returns
#     -------
#     skel : meshparty.skeleton.Skeleton object
#         origional object with all axon segments removed, saved to cloud

#     '''

#     # set up cloud path 
#     cloud_root = os.environ.get('SAVE_LOCATION', 'gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons')
#     cloud_path = os.path.join(cloud_root, mws_cf_destination_path)

#     classification_df_rf2 = make_classification_df(mw, m1, m2)

#     # recreate the meshwork files to include only their dendrite classified segments
#     print('skeletonizing ' + str(mw.seg_id) + "...")

#     # create and apply mask
#     # create mesh mask 
#     print('creating mesh mask') 
#     dendrite_mask = _skel_dendrite_map(classification_df_rf2)
#     my_verts = mw.SkeletonIndex(dendrite_mask[dendrite_mask['tf dendrite'] == True]['skeleton index'])
#     print('applying mesh mask') 
#     print(dendrite_mask)
#     mw.apply_mask(my_verts.to_mesh_mask)

#     print('saving to cloud') 

#     neuron_io.write_meshwork_h5_to_folder(mw, cloud_path)

#     print(str(mw.seg_id) + " skeletonized and saved to cloud")

# def make_classification_df(mw, m1, m2):
#     '''creates the classficiation df after 2 rounds of axon identification with 2 models'''
#     # import the ML models
#     # random forest model 1: classify axon/dendrite round 1
#     rf1 = joblib.load(m1)
#     # random forest model 2: classify axon dendrite with round 1 neighboring segment classification as feature
#     rf2 = joblib.load(m2)
#    # extract features
#     print('extracting features for body ' + str(mw.seg_id) + '...')
#     features_df = extract_features(mw)

#     # apply rf1 to features df
#     print('applying model 1...')
#     X1 = features_df.drop(['soma_pt', 'root_id', 'soma_id', 'segment', 'pre', 'post', 'pre_size', 'post_size', 'radius'], axis = 1)
#     predicted_segment_classifications_rf1 = rf1.predict(X1)
#     # save results to memory
#     classification_df_rf1 = features_df.copy()
#     classification_df_rf1['classification'] = predicted_segment_classifications_rf1

#     # add columns indicating neighboring segment classification for each segment
#     print('adding neighboring segment classification features...')
#     classification_df_rf1_neighbors = neighboring_segments(mw.skeleton, classification_df_rf1)
#     # priblem - I need the skeleton itself here... mayeb I can pull from google cloud

#     # apply rf2 to features df
#     print('applying model 2...')
#     X2 = classification_df_rf1_neighbors.drop(['classification', 'soma_pt', 'root_id', 'soma_id', 'segment', 'pre', 'post', 'pre_size', 'post_size', 'radius'], axis = 1)
#     predicted_segment_classifications_rf2 = rf2.predict(X2)
#     # save results to memory
#     classification_df_rf2 = classification_df_rf1_neighbors.copy()
#     classification_df_rf2['predicted classification rf2'] = predicted_segment_classifications_rf2
#     return classification_df_rf2


# @queueable
# def remove_axons_tq(source_cloud_folder, filename, destination_cloud_file, m1, m2):
    
#     '''
#     Parameters
#     ----------
#     cf: cloudfiles.cloudfiles.CloudFiles
#         the cloudfiles object that contains the folder in which 'filename'
#         is stored 
#         i.e. 
#         cf = CloudFiles(axon_id.neuron_io.add_cloud_path('cf_source_folder'))
#     filename : str
#         the name of the file in cf to have its axons removed
#     destination_cloud_file : str
#         the folder in the cloud path specified in .env.docker
#     m1 : str
#         path to the first ml model to remove axons
#     m2 : str
#         path to the second ml model to remove axons
    
#     '''
#     cf = CloudFiles(neuron_io.add_cloud_path(source_cloud_folder))
#     # cloadfiles removes the filename as bytes
#     # read those bytes into a BytesIO obj
#     with io.BytesIO(cf.get(filename)) as f:
#         # define the starting point as the first element 
#         f.seek(0)
#         # load those bytes as mw obj
#         mw = meshwork.load_meshwork(f)
        
#         # remove axons from that obj
#         models.remove_axons(mw, destination_cloud_file, m1, m2)