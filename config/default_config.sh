# Scene scale foe NGP reconstruction
CONFIG_AABB_SCALE=4

# Target number of frames to extract from video
CONFIG_FRAME_COUNT=125 

# Path to directory containing object masks (relative to scene path)
CONFIG_MASK_DIR=input_masks

# Number of NGP optimization steps
CONFIG_N_STEPS=5000 

# Path to NGP repository (use $SOURCE_PATH to refer to toolkit repository root dir)
CONFIG_NGP_PATH=$SOURCE_PATH/submodules/instant-ngp

# Filename of scene video (relative to scene path)
CONFIG_VIDEO_NAME=input.mp4

# Depth tolerance (in ~mm) for generating visibility masks
CONFIG_VISIBILITY_TOLERANCE=25

# Flag to indicate whether this is a reference mesh
CONFIG_IS_REFERENCE=0

# Scale factor for the scene
CONFIG_SCALE_FACTOR=0

# Reference mesh for aligning/scaling across scenes
CONFIG_REFERENCE_SCENE=

# BOP object/mesh ID
CONFIG_BOP_ID=0

# Use Poisson mesh reconstruction from NGP depth maps
CONFIG_USE_POISSON_MESH=0
