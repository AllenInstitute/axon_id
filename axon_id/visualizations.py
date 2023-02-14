import vtk

import cloudvolume
from caveclient import CAVEclient
from meshparty import trimesh_vtk

#client = CAVEclient('minnie65_phase3_v1')
#cv = cloudvolume.CloudVolume(client.info.segmentation_source(), progress = False, use_https = True, parallel=24)


def visualize(msh, skel_color = (0,0,0), skel_width = 3,
              pre = True, pre_color = (0,0,.8), pre_size = 2000, 
              post = True, post_color = (0.9,.3,.7), post_size = 2000, 
              branch = True, branch_color = (.6,.5,.5), branch_size = 5000,
              end = True, end_color = (.1,.5,.1), end_size = 5000):
    
    """
    returns list of actors to create a VTK visualization of a skeleton. Optionally  
    includes specified points (presynaptic sites, postsynaptic sites, branch points,
    and end points). Optionally specify color and size of each. 
    
    If color/size are not specified, the defaults are:
        skeleton: black, line width = 3
        postsynaptic sites: dark pink, small (2000)
        presynaptic sites: dark blue, small (2000)
        end points of branches: green, large (5000)
        branching points: light pink, large (5000)
    
    Parameters
    ----------
    msh : meshparty.meshwork.meshwork.Meshwork
        meshwork object with skeleton based on the level 2 graph. See pcg_skel documentation for details.
        must define either meshwork or root_id.
    skel_color : tuple 
        a len(3) tuple with the 0-1 rgb color of the skeleton
    skel_width : int
        intager that defines the thickness of the skeleton visualization
    pre : bool
        boolean indicating whether or not to return presynaptic sites visualized
    pre_color : tuple
        a len(3) tuple with the 0-1 rgb color of the presynaptic sites
    pre_size : int
        intager that defines the size of the presynaptic sites 
    post : bool
        boolean indicating whether or not to return postsynaptic sites visualized
    post_color : tuple
        a len(3) tuple with the 0-1 rgb color of the postsynaptic sites
    post_size
        intager that defines the size of the postsynaptic sites 
    branch : bool
        boolean indicating whether or not to return branch points visualized
    branch_color : tuple
        a len(3) tuple with the 0-1 rgb color of the branch points
    branch_size : int
        intager that defines the size of the branch points
    end : bool
        boolean indicating whether or not to return endpoints visualized
    end_color : tuple
        a len(3) tuple with the 0-1 rgb color of the endpoints
    end_size
        intager that defines the size of the endpoints

    
    Returns
    -------
    actors_list : list
        list of actors to enter into meshparty.trimesh_vtk.render_actors(actors_list)
        to visualize the neuron
    
    """
    
    
    
    # create skeleton actor
    skel = trimesh_vtk.skeleton_actor(msh.skeleton, color = skel_color, line_width = skel_width)

    
    actors_list = [skel]
    
    
    if pre == True:
        # add presynaptic sites actor to be added to the visualization
        presyn_actor = trimesh_vtk.point_cloud_actor(msh.anno.pre_syn.points, size = pre_size, 
                                                               color = pre_color)
        actors_list.append(presyn_actor)
        
        
    if post == True:
        # add postsynaptic sites actor
        postsyn_actor = trimesh_vtk.point_cloud_actor(msh.anno.post_syn.points,size = post_size, 
                                                                color = post_color)
        actors_list.append(postsyn_actor)
    
    if branch == True:
        # create branch points actor  
        branchpoints_actor = trimesh_vtk.point_cloud_actor(msh.skeleton.vertices[msh.skeleton.branch_points], 
                                                                     size = branch_size, color = branch_color) 
        actors_list.append(branchpoints_actor)
    
    if end == True:
        # create endpoints actor
        endpoints_actor = trimesh_vtk.point_cloud_actor(msh.skeleton.vertices[msh.skeleton.end_points], 
                                                                  size = end_size, color = end_color) 
        actors_list.append(endpoints_actor)

    
    # put it all together to visualize w/ vtk
    return actors_list   
    
def visualize_segment(msh, segment, skel_color = (0,0,0), skel_width = 3,
              pre = True, pre_color = (0,0,.8), pre_size = 2000, 
              post = True, post_color = (0.9,.3,.7), post_size = 2000, 
              branch = True, branch_color = (.6,.5,.5), branch_size = 5000,
              end = True, end_color = (.1,.5,.1), end_size = 5000):
    
    """
    creates a neuron visualization and focuses the camera on a given neuron segment
    
    
    Parameters
    ----------
    msh : meshparty.meshwork.meshwork.Meshwork
        Meshwork object with skeleton based on the level 2 graph. See pcg_skel documentation for details.
    segment : list of skeleton indices
        a path from a branch or end point (inclusive) to the next rootward branch/root point (exclusive), 
        that cover the skeleton. The visualization will be centered at this segment. 
    skel_color : tuple 
        a len(3) tuple with the 0-1 rgb color of the skeleton
    skel_width : int
        intager that defines the thickness of the skeleton visualization
    pre : bool
        boolean indicating whether or not to return presynaptic sites visualized
    pre_color : tuple
        a len(3) tuple with the 0-1 rgb color of the presynaptic sites
    pre_size : int
        intager that defines the size of the presynaptic sites 
    post : bool
        boolean indicating whether or not to return postsynaptic sites visualized
    post_color : tuple
        a len(3) tuple with the 0-1 rgb color of the postsynaptic sites
    post_size
        intager that defines the size of the postsynaptic sites 
    branch : bool
        boolean indicating whether or not to return branch points visualized
    branch_color : tuple
        a len(3) tuple with the 0-1 rgb color of the branch points
    branch_size : int
        intager that defines the size of the branch points
    end : bool
        boolean indicating whether or not to return endpoints visualized
    end_color : tuple
        a len(3) tuple with the 0-1 rgb color of the endpoints
    end_size
        intager that defines the size of the endpoints
    
    Returns
    -------
    visualization : vtkmodules.vtkRenderingOpenGL2.vtkOpenGLRenderer
        visualization of the skeleton based on the level 2 graph and data points.
        focal point is the center of the given segment
    
    """
    # TO DO MAYBE: create a neuroglancer link centerd at this same centerpoint?
        # for double checking
    
    # focus the vtk camera on the central point of the segment
    ctr_point = msh.skeleton.vertices[segment[len(segment)//2]]
    camera = vtk.vtkCamera()
    camera.SetFocalPoint(ctr_point)

    
    # create actors
    actors_list = visualize(msh, skel_color = skel_color, skel_width = skel_width,
              pre = pre, pre_color = pre_color, pre_size = pre_size, 
              post = post, post_color = post_color, post_size = post_size, 
              branch = branch, branch_color = branch_color, branch_size = branch_size,
              end = end, end_color = end_color, end_size = end_size)
    
    # visualize
    visualization = trimesh_vtk.render_actors(actors_list, camera = camera)

    return visualization
    
def visualize_skeleton(skel, skel_color = (0,0,0), skel_width = 3, segment = None, 
              branch = False, branch_color = (.6,.5,.5), branch_size = 5000,
              end = False, end_color = (.1,.5,.1), end_size = 5000, 
              seg_center_display = False, seg_ctr_scale = (1000, 1000, 1000),
              seg_ctr_value = 'idx'):
    
    """
    creates a skeleton visualization with optional inclusions 
    
    Parameters
    ----------
    skel : meshparty.skeleton.Skeleton
        Skeleton object 
    skel_color : tuple 
        a len(3) tuple with the 0-1 rgb color of the skeleton
    skel_width : int
        intager that defines the thickness of the skeleton visualization
    segment : list of skeleton indices
        a path from a branch or end point (inclusive) to the next rootward branch/root point (exclusive), 
        that cover the skeleton. Optionally add to center the visualization on the center of this segment. 
    branch : bool
        boolean indicating whether or not to return branch points visualized
    branch_color : tuple
        a len(3) tuple with the 0-1 rgb color of the branch points
    branch_size : int
        intager that defines the size of the branch points
    end : bool
        boolean indicating whether or not to return endpoints visualized
    end_color : tuple
        a len(3) tuple with the 0-1 rgb color of the endpoints
    end_size : int
        intager that defines the size of the endpoints
    seg_center_display : bool
        boolean indicating whether or not to display the skeleton index/vertex of each segment center on that segment
        as a vtk follower
    seg_ctr_scale : tuple
        tuple giving the size of the vtk follower teext displaying each segment centerpoint 
    seg_ctr_value : str
        string indicating weather to display the skeleton index or vertex point of each segment center. 
        'vtx' for vertex, 'idx' for index 

    
    Returns
    -------
    visualization : vtkmodules.vtkRenderingOpenGL2.vtkOpenGLRenderer
        visualization of the skeleton and data points.
    
    """
    # TO DO MAYBE: create a neuroglancer link centerd at this same centerpoint?
        # for double checking
    
    # add assertions?
    if (segment is None) or (type(segment) == list):
        None
    else:
        raise TypeError('segment must be either None or a list of skeleton indices')


    # focus the vtk camera on the central point of the segment if the segment has been given
    if segment != None:
        ctr_point = skel.vertices[segment[len(segment)//2]]
        camera = vtk.vtkCamera()
        camera.SetFocalPoint(ctr_point)
    else:
        camera = None

    
    # create actors
    skel_actor = trimesh_vtk.skeleton_actor(skel, color = skel_color, line_width = skel_width)

    
    actors_list = [skel_actor]
    
    if branch == True:
        # create branch points actor  
        branchpoints_actor = trimesh_vtk.point_cloud_actor(skel.vertices[skel.branch_points], 
                                                                     size = branch_size, color = branch_color) 
        actors_list.append(branchpoints_actor)
    
    if end == True:
        # create endpoints actor
        endpoints_actor = trimesh_vtk.point_cloud_actor(skel.vertices[skel.end_points], 
                                                                  size = end_size, color = end_color) 
        actors_list.append(endpoints_actor)
    
    segments_list = list(skel.segments)
    if seg_center_display == True:
        for i in range(len(segments_list)):
            
            # find center point of each segment (skeleton index and vertex of the center point)
            seg = segments_list[i]
            seg_ctr_idx = seg[len(seg)//2] # skel index in the center of the segment
            seg_ctr_vtx = (skel.vertices[int(seg_ctr_idx)]) # 3d vertex at that skel idx location 
            
            # create the text actor for that seg center point
            text = vtk.vtkVectorText()
            if seg_ctr_value == 'idx':
                text.SetText(str(seg_ctr_idx)) # text displays the skeleton index
            elif seg_ctr_value == 'vtx':
                text.SetText(str(seg_ctr_vtx)) # text displays the vertex point
            else:
                raise TypeError('seg_ctr_value must either be \'idx\' or \'vtx\'')
            
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text.GetOutputPort())
            
            text_actor = vtk.vtkFollower()
            text_actor.GetProperty().ShadingOff()
            text_actor.SetMapper(text_mapper)
            text_actor.SetScale(seg_ctr_scale) # size set by function parameter 
            text_actor.AddPosition(seg_ctr_vtx)
            
            actors_list.append(text_actor)

    
    # visualize
    visualization = trimesh_vtk.render_actors(actors_list, camera = camera)

    return visualization

# from Forrest
import pandas as pd
import io
import os
from meshparty import skeleton, skeleton_io
SWC_COLUMNS = ('id', 'type', 'x', 'y', 'z', 'radius', 'parent',)
COLUMN_CASTS = {
    'id': int,
    'parent': int,
    'type': int
}
def apply_casts(df, casts):

    for key, typ in casts.items():
        df[key] = df[key].astype(typ)

        
def read_skeleton(root_id, nuc_id, cloud_path = "https://storage.googleapis.com/allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/BIL_neurons/file_groups/"
):
    file_path=cloud_path + f"{root_id}_{nuc_id}/{root_id}_{nuc_id}.swc"
    #print(file_path)
    df =read_swc(file_path)
    verts = df[['x','y','z']].values
    edges = df[['id','parent']].iloc[1:].values
    sk=skeleton.Skeleton(verts, edges, vertex_properties={'radius':df['radius'], 
                                                       'compartment':df['type']})
    return sk

def read_swc(path, columns=SWC_COLUMNS, sep=' ', casts=COLUMN_CASTS):

    """ Read an swc file into a pandas dataframe
    """
    if "://" not in path:
        path = "file://" + path

    #cloudpath, file = os.path.split(path)
    #cf = CloudFiles(cloudpath)
    #path = io.BytesIO(cf.get(file))
    
    df = pd.read_csv(path, names=columns, comment='#', sep=sep)
    apply_casts(df, casts)
    return df



import seaborn as sns
def plot_cell(ax, sk, title='', plot_depths=False):
    
    MORPH_COLORS = {3: "firebrick", 4: "salmon", 2: "steelblue"}
    
    for compartment, color in MORPH_COLORS.items():
        lines_x = []
        lines_y = []
        guess = None
        skn=sk.apply_mask(sk.vertex_properties['compartment']==compartment)
        for cover_path in skn.cover_paths:
            path_verts = skn.vertices[cover_path,:]
            ax.plot(path_verts[:,0], path_verts[:,1], c=color, linewidth=1)
            
        ax.set_aspect("equal")
    plt.gca().invert_yaxis()

    #ax.set_ylim(1100, 300)
    sns.despine(left=True, bottom=True)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_title(title)

# for k, row in select_df[['pt_root_id','id']].iterrows():

#     sk = read_skeleton(row.pt_root_id, row.id)
#     f, ax = plt.subplots(figsize=(5,5))
#     plot_cell(ax, sk)
#     if k>10:
#         break
