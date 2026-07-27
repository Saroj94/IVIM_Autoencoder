import torch 
import numpy as np

##data extraction function

def similar_neighborhood(mri_data, brain_mask, b0_threshold=1.0):
    """
    Similar neighborhoods search:
    mri_volume: 4D numpy array shape (Num_b, Depth, Height, Width)
    brain_mask: 3D binary array of shape (Depth, Height, Width) where 1=brain, 0=outside
    b0_threshold: Minimum signal value(b0) to classify a voxel as valid brain tissue

    fused_block: numpy array of shape (N,3,3,Num_b) containing the 3x3 neighboring voxels
    center_coords: List of length N mapping each block  back into its original (x,y,z) position 
    """
    ##rearrange the dimension
    mri_volume = np.transpose(mri_data, (3,2,0,1))

    ##extracting num_b, depth, height, width
    b_value,D,H,W = mri_volume.shape

    ##normalization factors
    z_factor = 2.0/(D-1) if D>1 else 1.0
    y_factor = 2.0/(W-1) if W>1 else 1.0
    x_factor = 2.0/(H-1) if H>1 else 1.0

    ##extract b0 index - the first image in the sequence
    b0_image = mri_volume[0,:,:,:]

    ##find the coordinates of all vaild voxels inside the brain region 
    ## by combining the structural mask and minimal signal, filter out the empty air pixels outside brain
    brain_indices = np.argwhere((brain_mask>0) & (b0_image>=b0_threshold))
    
    ##N: total number of valid voxels to process
    N = len(brain_indices)

    ##To extract a 5x5 window , we must look 2 pixels out in all directions
    ##Padding the edges to avoid the index error
    padded_volume = np.pad(mri_volume, ((0,0),(2,2),(2,2),(2,2)), mode='constant', constant_values=0)

    ##first padded signal from the padded volume of mri data
    ##first b-vaue (b=0) slice
    padded_b0 = padded_volume[0,:,:,:]

    ##initialize the output array where N block of size 3x3 accross b-value channels
    similar_block = np.zeros((N, 3, 3, b_value), dtype=np.float32)

    ##center coordination
    center_coords = []

    ##normalize center coord
    center_norm_coord = []

    print(f"Processing {N} brain voxels to build the similarity matrics")
    
    for idx, (z,y,x) in enumerate(brain_indices):
        ##shift the coordinates by +2 to align with newly padded space
        pz, py, px = z+2, y+2, x+2

        ##current target central voxel signal on b0
        central_b0_signal = padded_b0[pz, py, px]

        ##window of 5x5 surrounding on b0 image from center-2 and center+3 (indics slicing)
        b0_neighbours = padded_b0[pz-2:pz+3, py-2:py+3, px-2:px+3]

        ##selected neighbours and similarity metrics
        neighbours_list=[]

        ##calculate the similar distance accross 5x5 window spatial context
        for nz in range(5):
            for ny in range(5):
                for nx in range(5):
                    ##exclude the center voxel from the search
                    if nz==2 and ny==2 and nx==2:
                        continue
                    ##fetch the intensity of the voxels at b0 
                    b0_neighbours_signal = b0_neighbours[nz, ny, nz]

                    ##similar signal euclidian distance: S_similar=sqrt((S_i - S_j)^2)
                    ##similarity = absolute difference on b0
                    s_similar = np.sqrt((central_b0_signal - b0_neighbours_signal)**2)

                    ##similar neighbours appeded
                    neighbours_list.append({'similarity':s_similar,
                                            'rel_coord':(nz, ny, nx)
                                            })

        ##8 most similar neighbours
        neighbours_list.sort(key=lambda x:x['similarity'])

        ##top8 similar neigbhours
        top8_neighbours = neighbours_list[:8]

        ##3x3 block accross all b-values
        full_padded_volume = padded_volume[:, pz-2:pz+3, py-2:py+3, px-2:px+3]

        ##initialize the empty 3x3 blocks for all channels
        block3 = np.zeros((3,3, b_value), dtype=np.float32)

        ##place the original central voxels exactly in the center of the new
        block3[1,1,:] = full_padded_volume[:, 2, 2, 2]

        ##flatten the 8 surrounding neighbourings
        #Coordinates exclude the central index slot (1,1)
        target_slot = [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]   

        ##distribute the selected 8 voxel signal configuration into the 3x3 spatial layout
        for slot_idx, neighbour in enumerate(top8_neighbours):
            nz, ny, nx = neighbour['rel_coord']
            ty, tx = target_slot[slot_idx]
            # Map the multi-b-value profile from the 5x5 source to the 3x3 target configuration
            block3[ty, tx, :] = full_padded_volume[:, nz, ny, nx]

        # Commit completed matrix to memory storage allocation
        similar_block[idx]=block3 
        center_coords.append((z,y,x))

        ##normalize coordinates for center voxel
        nz = z*z_factor-1.0
        ny = y*y_factor-1.0
        nx = x*x_factor-1.0
        ##normalize center coordinates
        center_norm_coord.append((nz,ny,nx))

    return similar_block, center_coords, center_norm_coord

print("code successfully executed")





