import pandas as pd
import numpy as np
import joblib


from axon_id import graph
from meshparty import meshwork 
            

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

    


# Put it all together

# function to take a mw, extract all features, then mask out the axons 

def class_1_2_axons(mw, m1, m2, model2_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
       'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad','classification', 'axon neighbor count', 
       'dendrite neighbor count', 'upstream_class'], 
       model1_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
        'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad'], 
        soma_seg_len = 10000, mask_out_ax = False, axon_class_anno = 'is_axon_class'):
    
    '''
    applies first and second classifier. saves axon verts in anno
    optionally applies mask to mask out axons 
    '''

    # create m1 and m2 classification df 
    class_df = classify_axon_dendrite_2(mw, m1, m2, model2_columns = model2_columns, model1_columns = model1_columns,
                        soma_seg_len = soma_seg_len) 

    # now get a map 
    skel_mask_df = graph.skel_dendrite_map(class_df, mw)
    skel_ax_verts = mw.SkeletonIndex(skel_mask_df[skel_mask_df['tf dendrite'] == False]['skeleton index'])
    mesh_ax_verts = skel_ax_verts.to_mesh_index
    # save the annotation
    mw.anno.add_annotations(axon_class_anno, mesh_ax_verts, mask=True)


    # apply mask to skeleton
    if mask_out_ax:
        mw.apply_mask(~mesh_ax_verts.to_mesh_mask)
    return mw


def find_primary_axon(mw, certainty_threshold = 0.7, pre_anno='pre_syn', post_anno='post_syn',
                            mask_out_ax = False, primary_axon_anno = 'is_primary_axon'):
    '''
    identifies primary axon, saves axon mesh segments in primary axon
    if certainty score is high enough, optionally masks out that id'd axon
    
    '''

    axon_inds, Q = meshwork.algorithms.split_axon_by_annotation(mw, pre_anno=pre_anno, post_anno=post_anno)

    if Q >= certainty_threshold:
        mw.anno.add_annotations(primary_axon_anno, axon_inds, mask=True)
    else:
        mw.anno.add_annotations(primary_axon_anno, mw.MeshIndex([]), mask=True)
    # mask the mw if Q ? certainty threshold 
    if Q >= certainty_threshold and mask_out_ax:
        mw.apply_mask(axon_inds)
    elif Q < certainty_threshold:
        print(f'certainty threshold not met, primary axon not masked out. Q = {Q}')

    return mw


def combine_primary_and_axon_class(mw, axon_class_anno = 'is_axon_class', 
                                    primary_axon_anno = 'is_primary_axon',
                                    ax_agglom_anno = 'is_all_axon',
                                    mask_out_ax = False):

    '''
    takes two axon labels and adds another annotation that has the set of both together 
    '''

    # find the entire set of all axon mesh indices 
    all_axon = mw.MeshIndex(list(set(mw.anno[axon_class_anno]['mesh_index']) | set(mw.anno[primary_axon_anno]['mesh_index'])))

    mw.anno.add_annotations(ax_agglom_anno, all_axon, mask=True)

    if mask_out_ax:
        mw.apply_mask(~mw.anno[ax_agglom_anno].mesh_mask)


def add_class_12_primary_anno(mw, m1, m2, model2_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
       'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad','classification', 'axon neighbor count', 
       'dendrite neighbor count', 'upstream_class'], 
       model1_columns = ['length', 'n_pre', 'pre_size', 'n_post', 'post_size', 'total_syn',
        'density', 'soma_dist', 'radius', 'endpoint', 'self/parent_rad'], 
        soma_seg_len = 10000, mask_out_ax = False, axon_class_anno = 'is_axon_class', 
        certainty_threshold = 0.7, pre_anno='pre_syn', post_anno='post_syn',
        primary_axon_anno = 'is_primary_axon', 
        ax_agglom_anno = 'is_all_axon'):
        '''
        returns two things, mw (with new annos and optionally applied mask), axon segment map

        apply round 1 and 2 classification, add annotation for level 2 axon class saved as axon_class_anno
        apply primary axon classification, add annotation as primary_axon_anno
        combine those two as ax_agglom_anno
        also return segment axon map
        '''
        # add class 1, 2 anno 
        class_1_2_axons(mw, m1, m2, model2_columns = model2_columns, 
            model1_columns = model1_columns, 
            soma_seg_len = soma_seg_len, mask_out_ax = mask_out_ax, axon_class_anno = axon_class_anno)
        
        # add primary axon anno 
        find_primary_axon(mw, certainty_threshold = certainty_threshold, pre_anno=pre_anno, 
                            post_anno=post_anno, mask_out_ax = mask_out_ax, primary_axon_anno = primary_axon_anno)
        
        # combine these two 
        combine_primary_and_axon_class(mw, axon_class_anno = axon_class_anno, 
                                    primary_axon_anno = primary_axon_anno,
                                    ax_agglom_anno = ax_agglom_anno,
                                    mask_out_ax = mask_out_ax)
        
        # now add axon segment map
        ax_seg_map = get_segment_axon_map(mw, ax_agglom_anno = ax_agglom_anno)

        return mw, ax_seg_map




def get_segment_axon_map(mw, ax_agglom_anno = 'is_all_axon'):
    '''
    creates a boolean map of whether or not each segment is axon or not
    '''
    
    # translate mesh to skeleton indices 
    axon_skinds = mw.anno[ax_agglom_anno].skel_index
    # search for last skeleton index or each segment
    seg_endpts = [seg[0] for seg in mw.skeleton.segments]
    # now get map of if each seg_endpt is in axon_skinds 
    tf_axon = [int(x) in axon_skinds for x in seg_endpts]

    return tf_axon


