import os
import meshparty
import pandas as pd
import pcg_skel
import cloudvolume
from caveclient import CAVEclient
from meshparty import meshwork
from meshparty.skeleton_io import write_skeleton_h5
from taskqueue import queueable
import io
import h5py
from cloudfiles import CloudFiles

client = CAVEclient('minnie65_phase3_v1')
cv = cloudvolume.CloudVolume(client.info.segmentation_source(), progress = False, use_https = True, parallel=24)

client.materialize.version = 117 

def write_meshwork_h5_to_cf(msh, cf_path, filename = None):

    """
    writes all parts of meshwork to an h5, saves in a specified path
    
    Parameters
    ----------
    msh : meshparty.meshwork.meshwork.Meshwork
        meshwork of the neuron to be saved
    cf_folder : string
        path to the folder in which you wish the meshes to be stored
        example - local file ./meshworks would be -  'file://./meshworks'

    """
    cf = CloudFiles(cf_path)

    if filename == None:
        filename = msh.seg_id

    with io.BytesIO() as bio:
        msh.save_meshwork(bio)
        bio.seek(0)
        cf.put(f'{filename}.h5', bio.read())

def write_skeleton_h5_to_cf(skel, cf_path, filename = None):

    """
    writes all parts of meshwork to an h5, saves in a specified path
    
    Parameters
    ----------
    msh : meshparty.meshwork.meshwork.Meshwork
        meshwork of the neuron to be saved
    cf_folder : string
        path to the folder in which you wish the meshes to be stored
        example - local file ./meshworks would be -  'file://./meshworks'

    """
    cf = CloudFiles(cf_path)

    if filename == None:
        filename = skel.meta.root_id

    with io.BytesIO() as bio:
        write_skeleton_h5(skel, bio)
        bio.seek(0)
        cf.put(f'{filename}.h5', bio.read())


def load_mws_from_cloud(cf_folder_path, asdict = False, update_roots = False):
    '''
    loads all meshwork files from a folder in the cloud
    cloud root is in .env.docker, cf_folder path will just be a folder name in the cloud location
    allen-minnie-phase3/minniephase3-emily-pcg-skeletons
    
    '''
    cloud_root = os.environ.get('SAVE_LOCATION', 'gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons')
    cloud_path = os.path.join(cloud_root, cf_folder_path)

    cf = CloudFiles(cloud_path)
    binaries = cf[:]

    # create a list or dict that will be filled with meshwork objects
    if asdict == True:
        mwdict = {}
    else:
        mwlist = []

    
    for i in range(len(binaries)):
        with io.BytesIO(cf.get(binaries[i]['path'])) as f:
            f.seek(0)
            mw = meshwork.load_meshwork(f)
            if update_roots == False:
                seg_id = mw.seg_id
            elif update_roots == True:
                seg_id = get_root_id_from_point(mw.skeleton.root_position)

            if asdict == True:
                mwdict[seg_id] = mw
            else:
                mwlist.append(mw)
            
    
    if asdict == True:
        return mwdict
    else:
        return mwlist



@queueable
def list_meshworks(bodies, folder_path, verbose = False, secret = None):
    
    """
    returns a list of meshworks from a list of supervoxel ids and stores those meshes in a specified local path
    
    Parameters
    ----------
    bodies : list
        list of supervoxel ids 
    folder_path : string
        path to the folder in which you wish the meshes to be stored in the format 
        seen here: https://github.com/seung-lab/cloud-files/#constructor
        example - local file ./meshworks would be -  'file://./meshworks'
    verbose : boolean, optional
        indicates weather or not to print the supervoxel and root ids of each file
    cache : str or None, optional
        Filename to a sqlite database with cached lookups for l2 ids. Optional, default is None.
   
    """
    
    
    if secret != None:
        cldvol = cloudvolume.CloudVolume(client.info.segmentation_source(), progress = False, 
        use_https = True, parallel=24, secrets = secret)
    else:
        cldvol = cv

    # pull out the table which contains soma centroid locations
    soma_df = client.materialize.query_table('nucleus_neuron_svm') 
    # move this ^ outside the function  

    count = 0
    for body in bodies:

        soma_location = soma_df[soma_df['pt_supervoxel_id'] == body]['pt_position'].iloc[0]
        # 

        # find the root id from the supervoxel id
        
        filename = folder_path.split('//')[-1] + "/" + str(body) + ".h5"

        # if the filename is already a file in the meshworks folder, load that mesh then skip the rest
        # this is going to be messed up with cloud files

        cloud_root = os.environ.get('SAVE_LOCATION', 'file://./axon_id/saved_meshworks')
        cloud_path = os.path.join(cloud_root, folder_path)
        if CloudFiles(cloud_path).exists(filename):
            msh = meshwork.load_meshwork(folder_path.split('//')[-1] + '/' + filename)
            continue

        root_id = int(client.chunkedgraph.get_roots(body))
        

        

        msh = pcg_skel.pcg_meshwork(root_id = root_id, client = client, cv = cldvol, refine = 'all', 
                                    synapses = 'all', collapse_soma=True, synapse_table = 'synapses_pni_2',
                                    root_point = soma_location, root_point_resolution = [4,4,40], 
                                    segmentation_fallback = False, )
        
        
        write_meshwork_h5_to_cf(msh, cloud_path, filename = body)
                 
        if verbose == True:
            print(str(count) + ': ' + str(body) + ', ' +  str(root_id))
            print('\n')
    
    


def retreive_meshes(folder_path):
    
    """
    loads all meshworks from a folder and outputs them as a list of meshwork objects
    
    Parameters
    ----------
    folder_path : string
        the path to the folder in which your meshes are stored
        i.e.'./meshworks'
    
    Returns
    -------
    mesh_list : list
        list of meshwork objects pulled from the defined folder_path
   
    """    
    
    mesh_list = []
    folder_name = folder_path.split("/")[-1]
    
    for filename in os.listdir(folder_path):
        
        if '.h5' in filename:
            
            filepath = folder_name + "/" + filename
            mesh_list.append(meshwork.load_meshwork(filepath))
    
    return mesh_list  
     

def root_to_supervoxel(root_id_list, soma_df = pd.DataFrame(), verbose = False):
    '''
    takes in a list of root ids and returns their supervoxel ids


    Parameters
    ----------
    root_ids : list
        list of root ids
    soma_df : pd.DataFrame, optional
        caveclient's 'nucleus_neuron_svm' table.
        soma_df = client.materialize.query_table('nucleus_neuron_svm')
        optioanlly put in as a parameter so that it does not have to be calculated every time if using this function in a loop 
    verbose : boolean, opional
        if true, will print root id = x, supervoxel id = y for each body in root list. 

    Returns
    -------
    supervoxel_ids : list
        list of supervoxel ids
    
    
    '''
    # if a soma_df was not entered 
    if len(soma_df) == 0:
        soma_df = client.materialize.query_table('nucleus_neuron_svm')

    supervoxel_ids_list = []
    for root_id in root_id_list:
        
        body_df = soma_df[soma_df['pt_root_id'] == root_id]['pt_supervoxel_id']
        
        # under 2% of neurons have multiple predicted somas. 
        # in that case, pick the 
        if len(body_df) > 1:
            pass

        supervoxel_id = soma_df[soma_df['pt_root_id'] == root_id]['pt_supervoxel_id'].iloc[0]
        supervoxel_ids_list.append(supervoxel_id)

        if verbose == True:
            print('root id = ' + str(root_id))
            print('supervoxel id = ' + str(supervoxel_id))

    return supervoxel_ids_list




def get_root_id_from_point(point, voxel_resolution = [1,1,1]):

    '''
    gets the currrent root_id for a root point
    point = mw.skeleton.vertices[mw.skeleton.root] if starting from mw
    skeleton root point must be set to the soma point 
    '''
    # get the current timestamp for the caveclient tables so that 
    # result will match current tables 
    timestamp = client.materialize.get_timestamp()

    return int(cv.download_point(point, size=1, coord_resolution=voxel_resolution, agglomerate=True, timestamp=timestamp))

def update_root_from_mw(mw):
    '''
    returns updated root based on the root point location in msh if possible. if no 
    root point location, updates via the update_root fn
    
    '''

    root_id = mw.seg_id

    if client.chunkedgraph.is_latest_roots([int(root_id)]):
        return root_id   

    elif str(mw.skeleton.root_position) != '[nan nan nan]':
        return get_root_id_from_point(mw.skeleton.root_position)
    
    else:
        raise ValueError('unable to get root position from meshwork')


# this does not work! when two bodies are split from one and when there is cutting in proofreading
# def update_root(root_id):

#     '''
#     takes in a root_id and returns the most recent root id for that neuron
    
#     '''

#     if client.chunkedgraph.is_latest_roots([int(root_id)]):
#         return root_id

#     else:
#         # get lineage graph, which tracks the root_id as it has changed 
#         # as the body was cut/combined with other bodies
#         lineage_dict = client.chunkedgraph.get_lineage_graph(root_id)['links']

#         # create a list to store all the target root ids and a list to store source root ids
#         source_list = []
#         target_list = []

#         for row in lineage_dict:
#             target_list.append(row['target'])
#             source_list.append(row['source'])
        
#         return list((set(source_list) ^ set(target_list)) & set(target_list))[0]



@queueable
def pcg_skeletonize(root_id, cloud_path):
    mw = pcg_skel.pcg_meshwork(root_id = root_id, client = client, cv = cv, refine = 'all')
    write_meshwork_h5_to_cf(mw, cloud_path)



def add_cloud_path(could_folder):

    '''
    takes in cloud folder and joins it with the save location specified in env.docker
    to return the full cloud file path to be used with cloud-files 
    
    
    '''    
    
    cloud_root = os.environ.get('SAVE_LOCATION', 'gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons')
    cloud_path = os.path.join(cloud_root, could_folder)
    return cloud_path