import pandas as pd
from itertools import combinations

def parent_rad(seg, feat_df, upstream_df, soma_rad = 8000):
    # finds the radius of a parent segment
    
    # get parent from upstream_df
    my_rad = feat_df.iloc[seg]['radius']
    try:
        upseg = upstream_df[upstream_df['downstream_seg'] == seg]['upstream_seg'].iloc[0]
        parent_radius = feat_df.iloc[upseg]['radius']
    except:
        # parent is soma
        parent_radius = soma_rad
    return round(my_rad/parent_radius, 3)

def parent_class(seg, class_df, downstream_segment_graph):

    try: 
        # look up the parent of the seg 
        upstream_seg = downstream_segment_graph[downstream_segment_graph['downstream_seg'] == seg]['upstream_seg'].iloc[0]

        # get the class of that seg 
        upstream_class = class_df.iloc[upstream_seg]['classification']
    except: # body is soma, no upstream class, call it dendrite (for keep/remove purposes)
        upstream_class = 0

    return upstream_class

def downstream_segment_graph(sk, columns = ['upstream_seg', 'downstream_seg'],
                                sortcol = 'upstream_seg'):
    '''
    creates a df with column:
    'a' - the segment index of the first segment of an edge.
    'class a' - the class of the first segment.
    'b' - the segment index of the second segment of an edge.
    'class b' - the class of the second segment.

    '''

    seg_connectivity_df = pd.DataFrame(columns = columns)

    for skidx in sk.branch_points:

        segment_idx = sk.segment_map[skidx]

        child_nodes = sk.child_nodes(skidx)
        for node in child_nodes:
            child_seg_idx = sk.segment_map[node]
            new_row = [segment_idx, child_seg_idx]
            seg_connectivity_df.loc[len(seg_connectivity_df)] = new_row


    seg_connectivity_df.drop_duplicates(inplace = True)
    seg_connectivity_df.reset_index(inplace = True, drop = True)
    seg_connectivity_df.sort_values(by=[sortcol], inplace = True)
    
    return seg_connectivity_df

def skel_dendrite_map(df, mw, r2_class_col = 'r2_classification'):

    '''
    takes in features df, returns a map of all skeleton indices and boolean of weather or not that
    index is on a dendrite classified segment 

    '''
    
    map_df = pd.DataFrame(columns = ['skeleton index', 'tf dendrite'])
    tf_dendrite_dict = {0:True, 1:False}
    
    for i in range(len(df)):
        segment = df.loc[i, 'segment']
        tf_dendrite = tf_dendrite_dict[df.loc[i, r2_class_col]]
        
        for node in segment:
            map_df.loc[int(node), 'skeleton index'] = int(node)
            map_df.loc[int(node), 'tf dendrite'] = tf_dendrite
    
    map_df.sort_values(['skeleton index'], inplace=True)
    map_df.reset_index(inplace=True, drop=True)
    
    # make sure soma is not masked out 
    somaind = map_df[map_df['skeleton index'] == mw.skeleton.root].index
    print(somaind)
    map_df.loc[somaind, 'tf dendrite'] = True

    return map_df


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
            
            new_row = [int(ctrpt), segment_class, int(child_ctrpt), child_class]
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
    adds 'axon neighbor count' and 'dendrite neighbor count' columns 
    
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