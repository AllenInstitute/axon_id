import pandas as pd
import numpy as np
import vtk
import os


import cloudvolume
from caveclient import CAVEclient
from meshparty import meshwork, trimesh_vtk
from taskqueue import queueable

from . import visualizations
from . import neuron_io

client = CAVEclient('minnie65_phase3_v1')
cv = cloudvolume.CloudVolume(client.info.segmentation_source(), progress = False, use_https = True, parallel=24)


def find_axon_heuristic(msh, client = client, ratio_threshold = .8): # newer one
        
    """
    Identifies segments on a neuron that are likely axons (heuristic)
    
    
    Parameters
    ----------
    msh : meshparty.meshwork.meshwork.Meshwork
        Meshwork object with skeleton based on the level 2 graph. See pcg_skel documentation for details.
    client : caveclient.CAVEclient
        CAVEclient for a datastack
    ratio_threshold: float
        ratio of presynaptic sites/total synapses a segment has to classify
        the segment as an axon. 0.8 by default.
    
    Returns
    -------
    
    segments_on_axon : list
        list a list of segment ids that are likely axon
        a segment is a path from a branch or end point (inclusive) to the next rootward 
        branch/root point (exclusive), that cover the skeleton
    pre_not_axon : list
        list of list of segment ids that have at least 1 presynaptic site, but does not
        meet the cutoff for axon classification. 
    no_syn : list
        list of list of segment ids that do not have any synapses on them.
    
    """


    # loop through segments, if the segment has a presyn site,
    # pull out all the other presyn and postyn on the segment
    # find ratio of pre/total syn
    pre_idx = msh.skeleton.mesh_to_skel_map[msh.anno.pre_syn.mesh_index] 
    post_idx = msh.skeleton.mesh_to_skel_map[msh.anno.post_syn.mesh_index]
    
    df = pd.DataFrame(columns = ['root_id', 'segment', 'n_pre', 'n_post', 'total', 'ratio', 
                      'classification', 'double check'])
    
    
    segments = msh.skeleton.segments
    df['segment'] = msh.skeleton.segments
    df['root_id'] = (msh.seg_id)
    
    for i in range(len(df)):
        
        # find the pre and post syn on that segment
        n_presyn = len(set(segments[i]) & set(pre_idx)) 
        n_postsyn = len(set(segments[i]) & set(post_idx)) 
        total = n_presyn+n_postsyn
        
        df.loc[i, 'n_pre'] = n_presyn # 'n_pre'
        df.loc[i, 'n_post'] = n_postsyn # 'n_post'
        df.loc[i, 'total'] = total # 'total'
        
        try:
            synratio = n_presyn/(n_presyn + n_postsyn)
        except:
            synratio = ''
        

        
        
        # add the segment to their respective lists where appropriate
        
        if (n_presyn + n_postsyn) == 0: # TO DO : check if this is connected to the soma / branches close to soma, could be true axon
            df.loc[i, 'ratio'] = '' # 'ratio'
            df.loc[i, 'classification'] = '0 syn' # classification
            df.loc[i, 'double check'] = True

    
        elif synratio > ratio_threshold: 
            df.loc[i, 'ratio'] = synratio # 'ratio'
            df.loc[i, 'classification'] = 'axon' # classification  
            if synratio < .9:
                df.loc[i, 'double check'] = True
            else:
                df.loc[i, 'double check'] = False

                        
        elif synratio > 0:
            df.loc[i, 'ratio'] = synratio # 'ratio'
            df.loc[i, 'classification'] = 'pre on dendrite' # classification 
            if synratio > .1:
                df.loc[i, 'double check'] = True
            else:
                df.loc[i, 'double check'] = False
            
            
        else:
            df.loc[i, 'ratio'] = synratio # 'ratio'
            df.loc[i, 'classification'] = 'dendrite' # classification    
            df.loc[i, 'double check'] = False # should add something that will make this True if under 3 synapses 
                                              # and connected to an axon segment
    
    return df

def check_segment_classifications(msh):

    """
    Visualize each given segment then asks the user if said segment has been classified correctly.
    Returns a dataframe with each segment and the user's input.
    Used in creating ground truth
    
    
    Parameters
    ----------
    msh : meshparty.meshwork.meshwork.Meshwork
        Meshwork object with skeleton based on the level 2 graph. See pcg_skel documentation for details.
    
    Returns
    -------
    visualization : vtkmodules.vtkRenderingOpenGL2.vtkOpenGLRenderer
        visualization of the skeleton based on the level 2 graph and data points
        Focal point is the center of the given segment
    df : pd.DataFrame
        data frame containing all the segments on this body with the specified classification
        and the input indicating if that classificaiton is correct or not, as determined by the user
    
    """


    # create the automatic segment classifications to double check where necessary 
    auto_seg_df = find_axon_heuristic(msh) # columns = ['root_id', 'segment', 'n_pre', 'n_post', 'total', 'ratio', 'classification', 'double check']

    # create data frame to be returned
    checked_df = pd.DataFrame(columns = ['root_id', 'segment', 'classification', 'classification correct?'])
    checked_df['segment'] = auto_seg_df['segment']
    checked_df['classification'] = auto_seg_df['classification']
    checked_df['root_id'] = [msh.seg_id]*len(checked_df) # updated to make it not multiply the actual number lol
    
    # loop through df, show vtk centered on the segment, 
    # ask if the segment has been classified correctly, 
    # record that response in 'classification correct?' column
    
    
    for i in range(len(checked_df)):  
        
        if auto_seg_df.loc[i, 'double check'] == False:
            checked_df.loc[i, 'classification correct?']  = 'y (automatic)'

        else:
            print('Body: ' + str(msh.seg_id) + ' Classification: ' + auto_seg_df.loc[i, 'classification'] + '.')
            visualizations.visualize_segment(msh, checked_df['segment'][i])
            
            checked_df.loc[i, 'classification correct?'] = input('Has this segment been classified accurately? y/n/m: ')

    # save locally here 
    df_filename = 'checked_seg_classification_h/' + str(msh.seg_id) + '_checked_seg_classification.csv'
    checked_df.to_csv(df_filename)

    return checked_df

def create_seg_color_map(df):

    """
    TO DO : generalize this and put it in the visualizations tab.
    Takes in a classified df and shows axon classsified segments as blue and dendrite classified segments as orange. 
    DF must include the following columns : 'true classification', 'segent'

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing segment and classificaiton information 
           
    Returns
    -------
    color_df : pd.DataFrame
        dataframe containing each node and its classification color 
    """
    
    #create a df to keep track of this all 
    color_df = pd.DataFrame(columns = ['skel index', 'class', 'color'])
    count = 0 # count to iterate through color_df
    
    for i in range(len(df)):
        
        # the color for all skidx from this row will be determined by the 'true classification'
        # column in this line 
        if df.loc[i, 'true classification'] == 'axon':
            color = [1, 1, 0.5] # dark blue
        elif df.loc[i, 'true classification'] == 'dendrite':
            color = [.1, .5, 1] # orange
        else:
            color = [0,0,.8] # red
          
        for skidx in (df.loc[i, 'segment']).replace('[','').replace(']','').split(): 
            color_df.loc[count, 'skel index'] = int(skidx)
            color_df.loc[count, 'class'] = df.loc[i, 'true classification']
            color_df.loc[count, 'color'] = color
            count += 1
    
    color_df = color_df.sort_values('skel index')
    color_df.reset_index(inplace = True, drop = True)
    return color_df
        
        
      

def check_classifications_visually(sup_id, seg_class_df = pd.DataFrame(), seg_class_df_path = '', return_msh_and_df = False):

    # NEED HELP WITH THIS AND THE WHOLE GROUNDTRUTH PIPELINE LOL


    """
    Parameters
    ----------
    sup_id : int
        supervoxel id of the neuron to have its classifications checked 
    seg_class_df : pd.DataFrame() (optional)
        df containing the segment classifications for the neuron of the above specified sup_id 
        if no seg_class_df is specified, searches for one in the specified folder path
    seg_class_df_path : string (optional)
        path to the folder containing the seg_class df, which must have the sup_id in the filename.
    return_msh_and_df : boolean

        
    Returns
    -------
    color_df : pd.DataFrame
        dataframe containing each node and its classification color 
    
    
    """
    
    root_id = client.chunkedgraph.get_root_id(sup_id) 
    
    # pull out the pcg_skel meshwork for this supervoxel id
    for filename in os.listdir('./meshworks'):
        if str(sup_id) in filename:
            msh = meshwork.load_meshwork('./meshworks/' + filename)
            break
     
    # pull out the segment classification csv for this supervoxel id, open it with excel!
    if len(seg_class_df) == 0:
        for filename in os.listdir(seg_class_df_path):
            if str(root_id) in filename:
                seg_class_df = pd.read_csv('final_checked_seg_classification_h/' + filename)
                os.system("open -a '/Applications/Microsoft Excel.app' './final_checked_seg_classification_h/'" + filename)
            
    # create the color map for the skeleton that will color segments based on classification
    # blue = axon, orange = dendrite, pink = other (maybe a segmentation error)
    color_df = create_seg_color_map(seg_class_df)

    
            
    skel = trimesh_vtk.skeleton_actor(msh.skeleton, vertex_data = list(color_df['color']), line_width = 3)
    actor_list = [skel] # will add on the centerpoint label followers
    
    for i in range(len(seg_class_df)):
        
        # find center point of each segment (skeleton index and vertex of the center point)
        seg = (seg_class_df.loc[i, 'segment']).replace('[','').replace(']','').split()
        seg_ctr_idx = seg[len(seg)//2] # skel index 
        seg_ctr_vtx = (msh.skeleton.vertices[int(seg_ctr_idx)])
        
        text = vtk.vtkVectorText()
        text.SetText(seg_ctr_idx)
        
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text.GetOutputPort())
        
        text_actor = vtk.vtkFollower()
        text_actor.GetProperty().ShadingOff()
        text_actor.SetMapper(text_mapper)
        text_actor.SetScale(1000, 1000, 1000)
        text_actor.AddPosition(seg_ctr_vtx)
        
        actor_list.append(text_actor)
            
    # visualize the body with each segment labeled by the skeleton index of its center point

    visualization = trimesh_vtk.render_actors(actor_list)

    # TO DO : i should probably just make this a separate function, it's useful. maybe link_skel_class
    if return_msh_and_df == True:
        return msh, seg_class_df


def process_input_df(df):
    '''

    takes in the 'classification correct?' column and returns the true classification 

    '''

    for i in range(len(df)):
        if df.loc[i, 'classification correct?'] == 'y' or df.loc[i, 'classification correct?'] ==  'y (automatic)':
            if df.loc[i, 'classification'] == 'pre on dendrite':
                df.loc[i, 'true classification'] = 'dendrite'
            else:
                df.loc[i, 'true classification'] = df.loc[i, 'classification']

        # if the cell has both axon and dendrite, that's a problem and needs user input to decide which one is the classification
        # if it has neither axon nor dendrite, the user also has to decide. hopefully it just indicates a spelling mistake
        elif (('axon' in df.loc[i, 'classification correct?']) and ('dendrite' in df.loc[i, 'classification correct?'])) or (('axon' not in  df.loc[i, 'classification correct?']) and ('dendrite' not in  df.loc[i, 'classification correct?'])):
            df.loc[i, 'true classification'] = input('Auto class is ' + df.loc[i, 'classification'] + '. Input is ' + df.loc[i, 'classification correct?'] + '   ...What is your best guess at the classification? ')

        elif 'axon' in df.loc[i, 'classification correct?']:
            df.loc[i, 'true classification'] = 'axon'

        elif 'dendrite' in df.loc[i, 'classification correct?']:
            df.loc[i, 'true classification'] = 'dendrite'

    return df

def extract_features_training(msh_folder_path, df_folder_path): # do this but with a folder full of dfs 
    
    '''
    takes in a folder that contains all the dataframes with segments and their classifications
    will extract the features for each segment and visulaize. 



    Parameters
    ----------
    folder_path : string
        path to the directory that contains the dataframes 
        i.e. a folder in the same folder as your code - folder_path = './folder_name_here'
  
    Returns
    -------
    xxx : yyy
        zzz
    
    
    '''
    
    #creating the df. type = np.float to allow user to run sns.pairplot later. 

    final_df = pd.DataFrame(data = None, columns = ['meshwork', 'root_id', 'supervoxel_id', 'soma_pt', 'segment', 'ctr_pt',  
                                              'length', 'pre', 'n_pre', 'post', 'n_post', 'total_syn', 'density',
                                              'soma_dist', 'classification']) 
                                              # df from all meshworks in meshworks list
                                              # dtype has to be float in order to do pairplots later
    
    soma_df = client.materialize.query_table('nucleus_neuron_svm')

    # open and read 
    for filename in os.listdir(df_folder_path):
        print(filename)
        
        #idk what this hidden file is but I don't want to delete it and mess sometime up 
        if filename == '.DS_Store':
            continue
        root_id = filename[34:-4] # the part of the filename that contains the root id
        supervoxel_id = neuron_io.root_to_supervoxel(root_id_list = [int(root_id)], soma_df=soma_df)

        #read the csv
        checked_class_df = pd.read_csv((df_folder_path + '/' + filename))

        body_df =pd.DataFrame(data = None, columns = ['meshwork', 'root_id', 'supervoxel_id', 'soma_pt', 'segment', 'ctr_pt', 
                                              'length', 'pre', 'n_pre', 'post', 'n_post', 'total_syn', 'density',
                                              'soma_dist', 'classification']) 
        
        

        body_df['segment'] = checked_class_df['segment']

        # need to do this because thesthe cells in this columns will contain a set
        body_df['pre'] = body_df['pre'].astype('object')
        body_df['post'] = body_df['post'].astype('object')


        # find the coresponding skeleton meshwork file                                
        for msh_filename in os.listdir(msh_folder_path):
            if str(supervoxel_id[0]) in msh_filename:
                msh = meshwork.load_meshwork(msh_folder_path + '/' + msh_filename)


        
        for i in range(len(body_df)):

            body_df.loc[i, 'meshwork'] = msh

            body_df.loc[i, 'root_id'] = root_id

            body_df.loc[i, 'supervoxel_id'] = supervoxel_id[0].astype(np.int64)

            body_df.loc[i, 'soma_pt'] = msh.root_skel[0]
            
            # segments lost their list 
            seg = list(map(int,(body_df.loc[i, 'segment']).replace('[','').replace(']','').split()))
            
            body_df.loc[i, 'ctr_pt'] = seg[len(seg)//2]
            
            body_df.loc[i, 'length'] = len(seg)
            
            body_df.at[i, 'pre'] = set(seg) & set(msh.skeleton.mesh_to_skel_map[msh.anno.pre_syn.mesh_index])
            body_df.loc[i, 'n_pre'] = len(body_df.loc[i, 'pre'])
            
            body_df.at[i, 'post'] = set(seg) & set(msh.skeleton.mesh_to_skel_map[msh.anno.post_syn.mesh_index])
            body_df.loc[i, 'n_post'] = len(body_df.loc[i, 'post'])
            
            body_df.loc[i, 'total_syn'] = body_df.loc[i, 'n_pre'] + body_df.loc[i, 'n_post']
            body_df.loc[i, 'density'] = body_df.loc[i, 'total_syn']/body_df.loc[i, 'length']
            
            body_df.loc[i, 'soma_dist'] = len(msh.path_between(int(msh.root_skel), msh.skeleton.segments[i][-1], return_as_skel = True))
            # add mesh density - check the screenshot, axons are less dense usually 
            
            # this should be an external function 
            # if body_df.loc[i, 'total_syn'] == 0:

            # add final classificaiton to the segment
            classification = checked_class_df.loc[i, 'true classification']
            class_dict = {'dendrite':0, 'axon': 1, 'other': 2}
            body_df.loc[i, 'classification'] = class_dict[classification]
            


            
            
        
        final_df = pd.concat([final_df, body_df.replace(np.nan, '-')])
        
    final_df.reset_index(inplace = True, drop = True)
        
    return final_df