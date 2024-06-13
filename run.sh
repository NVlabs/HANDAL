#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


##################################################
# Load helper functions
##################################################
SOURCE_PATH=$(dirname "$BASH_SOURCE")
source $SOURCE_PATH/scripts/utils.sh


##################################################
# Loop over arguments and parse flags
##################################################
SCENE_PATH=""
for arg in "$@"; do
    case $arg in
        -h|--help)
            echo "Usage: bash run.sh [SCENE_PATH] [OPTIONS]"
            echo "  SCENE_PATH: Path to scene directory (default: current directory)"
            echo "  OPTIONS:"
            echo "    --interactive: Run interactive tool to align canonical pose"
            echo "    --help: Print this help message"
            exit 0
            ;;
        --interactive)
            IS_INTERACTIVE=1
            ;;
        *)
            # if SCENE_PATH is empty and arg is not an option, set SCENE_PATH
            if [[ -z $SCENE_PATH ]] && [[ ! $arg == -* ]]; then
                SCENE_PATH=$arg
            # otherwise, print error
            else
                error "Unknown argument $arg. Use flag --help for usage."
            fi
            ;;
    esac
done

# If not scene is given, use the current directory
if [ -z $SCENE_PATH ]; then
    SCENE_PATH="."
fi
if [ ! -d $SCENE_PATH ]; then
    error "Unable to find scene directory `path $SCENE_PATH`. Exiting."
else
    info "Processing scene `path $SCENE_PATH` ..."
fi


##################################################
# Redirect stdout and stderr to log file
##################################################
mkdir -p $SCENE_PATH/logs
LOG_FILE=$SCENE_PATH/logs/$(date +%Y-%m-%d_%H-%M-%S).log
info "Logging to `path $LOG_FILE` ..."
# exec > >(tee -i $LOG_FILE)
# exec > >(tee -i /dev/tty | sed -r 's/\x1b\[[0-9;]*m?//g' > $LOG_FILE)
exec > >(tee -i /dev/tty | stdbuf -oL sed -r 's/\x1b\[[0-9;]*m?//g' > $LOG_FILE)
    # uses sed to remove ansi escape sequences before writing to log file
    #  and uses stdbuf to flush stdout/stderr when writing to log file


##################################################
# Load and print config variables
##################################################
header "Loading config variables"
source $SOURCE_PATH/config/default_config.sh
if [ ! -e $SCENE_PATH/config.sh ]; then
    info "Unable to find scene config file `path $SCENE_PATH/config.sh`. Using defaults."
else
    info "Loading config file `path $SCENE_PATH/config.sh` ..."
    source $SCENE_PATH/config.sh
fi

config_vars=$(compgen -A variable | grep CONFIG_)  # list all variables with CONFIG_ prefix
for var in $config_vars; do
    cmd "$var=`eval echo \\$$var`"
done
echo

##################################################
# Extract frames from video as images
##################################################
header "Extracting frames from video"
image_files=($SCENE_PATH/images_scene/*.jpg)
if [ -e $image_files ]; then
    info "Found existing frames in `path "$SCENE_PATH/images_scene/*.jpg"`. Skipping frame extraction."
elif [ ! -e $SCENE_PATH/$CONFIG_VIDEO_NAME ]; then
    error "Unable to find video `path $SCENE_PATH/$CONFIG_VIDEO_NAME`. Exiting."
else
    python $SOURCE_PATH/scripts/extract_sharpest_frames.py \
        $SCENE_PATH/$CONFIG_VIDEO_NAME \
        $SCENE_PATH/images_scene \
        --min_sharpness_window_size -1 \
        --max_frame_count $CONFIG_FRAME_COUNT \
        --image_ext jpg ||
    exit_with_error
fi


##################################################
# Generate COLMAP camera poses
##################################################
header "Generating camera poses via COLMAP"
image_files=($SCENE_PATH/images_scene/*.jpg)
for COLMAP_MATCHER in sequential exhaustive; do
    if [ -e $SCENE_PATH/transforms_scene.json ]; then 
        info "Found existing `path $SCENE_PATH/transforms_scene.json`. Skipping COLMAP."
        break
    elif [ -e $SCENE_PATH/transforms_scene_$COLMAP_MATCHER.json ]; then 
        info "Found existing `path $SCENE_PATH/transforms_scene_$COLMAP_MATCHER.json`. Skipping $COLMAP_MATCHER matcher."
    elif [ ! -e $CONFIG_NGP_PATH/scripts/colmap2nerf.py ]; then 
        error "Unable to find NGP script `path $CONFIG_NGP_PATH/scripts/colmap2nerf.py`. Exiting."
    elif [ ! -e $image_files ]; then 
        error "Unable to find images in `path $SCENE_PATH/images_scene/*.jpg`. Exiting."
    else 
        info "Running COLMAP with `path $COLMAP_MATCHER` matcher..."
        if [ ! -d $SCENE_PATH/colmap_$COLMAP_MATCHER ]; then 
            mkdir $SCENE_PATH/colmap_$COLMAP_MATCHER
        fi
        ( 
            CONFIG_NGP_PATH=$(realpath $CONFIG_NGP_PATH)
            cd $SCENE_PATH &&
            python $CONFIG_NGP_PATH/scripts/colmap2nerf.py \
                --run_colmap \
                --images images_scene \
                --out transforms_scene.json \
                --overwrite \
                --aabb_scale $CONFIG_AABB_SCALE \
                --colmap_matcher $COLMAP_MATCHER \
                --colmap_db colmap_$COLMAP_MATCHER/colmap.db \
                --text colmap_$COLMAP_MATCHER/colmap_text
        ) &&
        python $SOURCE_PATH/scripts/edit_transforms_file.py \
            $SCENE_PATH/transforms_scene.json \
            --sort_frames ||
        exit_with_error
    fi

    # ensure that COLMAP succeeded by verifying that the number 
    # of occurences of "transform_matrix" in the transforms file
    # is at least 90% of the number of images in images_scene/
    num_transform_matrices=$(grep -c "transform_matrix" $SCENE_PATH/transforms_scene.json)
    num_images=$(ls $SCENE_PATH/images_scene/ | wc -l)
    if [ $num_transform_matrices -lt $((num_images * 9 / 10)) ]; then
        # if matcher is sequential, try exhaustive
        if [ $COLMAP_MATCHER == "sequential" ]; then
            mv $SCENE_PATH/transforms_scene.json $SCENE_PATH/transforms_scene_sequential.json
            export $COLMAP_MATCHER=exhaustive
            info "COLMAP computed poses for only $num_transform_matrices/$num_images images with `path sequential` matcher."
            info "Retrying with COLMAP matcher `path exhaustive`..."
        # otherwise, exit
        else
            mv $SCENE_PATH/transforms_scene.json $SCENE_PATH/transforms_scene_exhaustive.json
            error "COLMAP computed poses for only $num_transform_matrices/$num_images images with `path exhaustive` matcher. Exiting."
        fi
    else
        info "COLMAP succeeded with matcher `path $CONFIG_COLMAP_MATCHER`."
        python $SOURCE_PATH/scripts/edit_transforms_file.py \
            $SCENE_PATH/transforms_scene.json \
            --remove_omitted_frames_from_dir $SCENE_PATH/images_scene ||
        exit_with_error
        break
    fi
done


##################################################
# Train NGP reconstruction of full scene
##################################################
header "Training NGP reconstruction of full scene"
if  [[ -e $SCENE_PATH/nerf_scene.ingp ]]
then 
    info "Found existing `path $SCENE_PATH/nerf_scene.ingp`. Skipping scene NeRF."
else 
    python $CONFIG_NGP_PATH/scripts/run.py  \
        --scene $SCENE_PATH/transforms_scene.json \
        --save_snapshot $SCENE_PATH/nerf_scene.ingp \
        --n_steps $CONFIG_N_STEPS || 
    exit_with_error
    # TODO add flags for training latents and extrinsics (and export optimized poses)
    # TODO fix video rendering and render a debug video
fi 


##################################################
# Create masked training images and transforms file
##################################################
header "Creating masked images and transforms file"
raw_mask_files=($SCENE_PATH/$CONFIG_MASK_DIR/*.png)
masked_image_files=($SCENE_PATH/images_object/*.png)
if [ -e $masked_image_files ] && [ -e $SCENE_PATH/transforms_object.json ]; then 
    info "Found existing `path $SCENE_PATH/images_object/*.png` and" \
        "`path $SCENE_PATH/transforms_object.json`. Skipping image masking."
elif [ ! -e $raw_mask_files ]; then 
    error "Unable to find masks in `path $SCENE_PATH/$CONFIG_MASK_DIR`. Exiting."
else
    python $SOURCE_PATH/scripts/mask_images.py \
        $SCENE_PATH/images_scene \
        $SCENE_PATH/$CONFIG_MASK_DIR \
        $SCENE_PATH/images_object \
        --standardized_masks_dir $SCENE_PATH/masks_object &&
    python $SOURCE_PATH/scripts/edit_transforms_file.py \
        $SCENE_PATH/transforms_scene.json \
        --output_file $SCENE_PATH/transforms_object.json \
        --set_aabb_scale 1 \
        --edit_frame_file_path "images_scene/" "images_object/" \
        --edit_frame_file_path ".jpg" ".png" ||
    exit_with_error
fi


##################################################
# Train NGP reconstruction of masked object
##################################################
header "Training NGP reconstruction of masked object"
if  [[ -e $SCENE_PATH/nerf_object.ingp ]]
then 
    info "Found existing `path $SCENE_PATH/nerf_object.ingp`. Skipping object NeRF."
else 
    python $CONFIG_NGP_PATH/scripts/run.py  \
        --scene $SCENE_PATH/transforms_object.json \
        --save_snapshot $SCENE_PATH/nerf_object.ingp \
        --n_steps $CONFIG_N_STEPS ||
    exit_with_error
    # TODO add flags for training extrinsics & latents
    # TODO fix video rendering and render a debug video
    # TODO what is "andrewg_hack"?
fi 


##################################################
# Export mesh from NGP object reconstruction
##################################################
header "Exporting rough, untextured mesh from NGP via marching cubes"
if [[ -e $SCENE_PATH/meshes/raw.ply ]]
then 
    info "Found existing `path $SCENE_PATH/meshes/raw.ply`. Skipping mesh export."
elif
    [[ ! -e $SCENE_PATH/nerf_object.ingp ]]
then
    error "Unable to find NGP snapshot `path $SCENE_PATH/nerf_object.ingp`. Exiting."
else
    info "Exporting marching cubes mesh from NGP snapshot `path $SCENE_PATH/nerf_object.ingp` ..."
    mkdir -p $SCENE_PATH/meshes &&
    python $CONFIG_NGP_PATH/scripts/run.py \
        --load_snapshot $SCENE_PATH/nerf_object.ingp \
        --marching_cubes_res 256 \
        --marching_cubes_density_thresh 1.0 \
        --save_mesh $SCENE_PATH/meshes/raw.ply ||
    exit_with_error
fi


##################################################
# Clean mesh
##################################################
header "Cleaning marching cubes mesh"
if [[ -e $SCENE_PATH/meshes/base.ply ]]
then 
    info "Found existing `path $SCENE_PATH/meshes/base.ply`. Skipping mesh cleaning."
elif
    [[ ! -e $SCENE_PATH/meshes/raw.ply ]]
then
    error "Unable to find marching cubes mesh `path $SCENE_PATH/meshes/raw.ply`. Exiting."
else
    info "Cleaning mesh `path $SCENE_PATH/meshes/raw.ply` ..."
    python $SOURCE_PATH/scripts/clean_mesh.py \
        $SCENE_PATH/meshes/raw.ply \
        $SCENE_PATH/meshes/base.ply ||
    exit_with_error
fi


##################################################
# Generate raw BOP annotations (unscaled and not in canonical pose)
##################################################
header "Generating initial BOP annotations"
if [[ -e $SCENE_PATH/bop_raw/scene_gt_initial.json ]] && \
    [[ -e $SCENE_PATH/bop_raw/scene_camera_initial.json ]]
then 
    info "Found existing initial BOP files in `path $SCENE_PATH/bop_raw`. Skipping."
elif
    [[ ! -e $SCENE_PATH/transforms_scene.json ]]
then
    error "Unable to find camera poses `path $SCENE_PATH/transforms_scene.json`. Exiting."
else
    info "Generating initial BOP files in `path $SCENE_PATH/bop_raw` ..."
    python $SOURCE_PATH/scripts/generate_gt_bop.py \
        $SCENE_PATH/bop_raw \
        --camera_poses_fn $SCENE_PATH/transforms_scene.json \
        --bop_object_id $CONFIG_BOP_ID \
        --output_fn_tag "_initial" ||
    exit_with_error
fi


##################################################
# Export depth images from NGP scene reconstruction
##################################################
header "Exporting depth images from NGP scene reconstruction"
depth_files=($SCENE_PATH/depth_scene/*.png)
if [[ -e $depth_files ]]
then 
    info "Found existing depth images in `path "$SCENE_PATH/depth_scene/*.png"`. Skipping depth export."
elif
    [[ ! -e $SCENE_PATH/nerf_scene.ingp ]]
then
    error "Unable to find NGP snapshot `path $SCENE_PATH/nerf_scene.ingp`. Exiting."
else
    info "Exporting depth images from NGP snapshot `path $SCENE_PATH/nerf_scene.ingp` to `path $SCENE_PATH/depth_scene` ..."
    mkdir -p $SCENE_PATH/depth_scene &&
    python $SOURCE_PATH/scripts/export_depth.py \
        $SCENE_PATH/nerf_scene.ingp \
        $SCENE_PATH/depth_scene \
        --ngp_root $CONFIG_NGP_PATH \
        --colorized_depth_dir $SCENE_PATH/depth_scene_color \
        --clip_distance 10.0 ||
    exit_with_error
fi


##################################################
# Apply texture to mesh using Open3D pipeline
##################################################
header "Generating textured mesh"
if [[ -e $SCENE_PATH/meshes/textured.ply ]] && \
    [[ -e $SCENE_PATH/meshes/colored.ply ]]
then 
    info "Found existing `path $SCENE_PATH/meshes/textured.ply`. Skipping textured mesh."
elif [[ ! -e $SCENE_PATH/bop_raw/scene_gt_initial.json ]] || \
    [[ ! -e $SCENE_PATH/bop_raw/scene_camera_initial.json ]] || \
    [[ ! -e $SCENE_PATH/images_scene/ ]] || \
    [[ ! -e $SCENE_PATH/depth_scene/ ]] || \
    [[ ! -e $SCENE_PATH/masks_object ]]
then 
    error "Unable to find required input files. Exiting."
else
    # generate Poisson mesh and apply texture
    if [[ $CONFIG_GENERATE_POISSON_MESH == 1 ]]
    then
        info "Found config flag `path '$CONFIG_USE_POISSON_MESH == 1'`."
        info "Generating textured poisson mesh and saving to `path $SCENE_PATH/meshes/textured.ply` ..."
        python $SOURCE_PATH/scripts/o3d_mesh_pipeline.py \
            $SCENE_PATH/meshes/textured.ply \
            --colored_mesh_output_fn $SCENE_PATH/meshes/colored.ply \
            --bop_pose_fn $SCENE_PATH/bop_raw/scene_gt_initial.json \
            --bop_camera_fn $SCENE_PATH/bop_raw/scene_camera_initial.json \
            --rgb_dir $SCENE_PATH/images_scene \
            --depth_dir $SCENE_PATH/depth_scene \
            --mask_dir $SCENE_PATH/masks_object \
            --max_views 50 ||
        exit_with_error
    else
        info "Found config flag `path '$CONFIG_USE_POISSON_MESH == 0'`."
        info "Applying texture to existing mesh and saving to `path $SCENE_PATH/meshes/textured.ply` ..."
        python $SOURCE_PATH/scripts/o3d_mesh_pipeline.py \
            $SCENE_PATH/meshes/textured.ply \
            --colored_mesh_output_fn $SCENE_PATH/meshes/colored.ply \
            --subdivision_iterations 2 \
            --texture_existing_mesh $SCENE_PATH/meshes/base.ply \
            --bop_pose_fn $SCENE_PATH/bop_raw/scene_gt_initial.json \
            --bop_camera_fn $SCENE_PATH/bop_raw/scene_camera_initial.json \
            --rgb_dir $SCENE_PATH/images_scene \
            --depth_dir $SCENE_PATH/depth_scene \
            --mask_dir $SCENE_PATH/masks_object \
            --max_views 50 ||
        exit_with_error
        # TODO can we downsample to the original vertices while keeping the same texture coordinates?
    fi
fi

##################################################
# Prompt for canonical pose
##################################################
header "Setting canonical pose"
if [[ -e $SCENE_PATH/canonical_pose.json ]] && \
    [[ -e $SCENE_PATH/meshes/canonical.ply ]]
then 
    info "Found existing `path $SCENE_PATH/canonical_pose.json` and `path $SCENE_PATH/meshes/canonical.ply`. Skipping."
elif [[ $IS_INTERACTIVE != 1 ]] 
then 
    info "Unable to find existing canonical pose `path $SCENE_PATH/canonical_pose.json`."
    error "Setting canonical pose requires interactive GUI tool. Use flag `path --interactive` to run. Exiting."
elif [[ ! -e $SCENE_PATH/meshes/textured.ply ]]
then 
    error "Unable to find textured mesh `path $SCENE_PATH/meshes/textured.ply`. Exiting."
else
    info "Running interactive tool to set canonical pose ..."
    python $SOURCE_PATH/scripts/manual_align.py \
        $SCENE_PATH/meshes/colored.ply \
        --output_mesh_fn $SCENE_PATH/meshes/canonical.ply \
        --output_transform_fn $SCENE_PATH/canonical_pose.json ||
    exit_with_error
fi


##################################################
# Prompt for GT scale
##################################################
header "Setting scale"
if [[ -e $SCENE_PATH/canonical_scale.json ]] && \
    [[ -e $SCENE_PATH/meshes/scaled.ply ]]
then 
    info "Found existing `path $SCENE_PATH/canonical_scale.json` and `path $SCENE_PATH/meshes/scaled.ply`. Skipping."
elif [[ $CONFIG_SCALE_FACTOR != 0 ]]
then 
    info "Found config setting `path $CONFIG_SCALE_FACTOR`. Setting scale to $CONFIG_SCALE_FACTOR."
    python $SOURCE_PATH/scripts/manual_scale.py \
        $SCENE_PATH/meshes/canonical.ply \
        $SCENE_PATH/meshes/scaled.ply \
        $SCENE_PATH/canonical_scale.json \
        --scale_factor $CONFIG_SCALE_FACTOR ||
    exit_with_error
elif [[ $IS_INTERACTIVE != 1 ]]
then 
    info "Unable to find existing canonical scale `path $SCENE_PATH/canonical_scale.json`."
    error "To set scale interactively, use flag `path --interactive` to run. Exiting."
elif [[ ! -e $SCENE_PATH/meshes/canonical.ply ]]
then 
    error "Unable to find textured mesh `path $SCENE_PATH/meshes/canonical.ply`. Exiting."
else
    info "Running interactive tool to set canonical scale ..."
    python $SOURCE_PATH/scripts/manual_scale.py \
        $SCENE_PATH/meshes/canonical.ply \
        $SCENE_PATH/meshes/scaled.ply \
        $SCENE_PATH/canonical_scale.json ||
    exit_with_error
fi

##################################################
# Apply scale and canonical pose to BOP annotations
##################################################
header "Applying canonical pose and scale"
if [[ -e $SCENE_PATH/bop_raw/scene_gt_canonical.json ]] && \
    [[ -e $SCENE_PATH/bop_raw/scene_camera_canonical.json ]]
then 
    info "Found existing output files in `path $SCENE_PATH/bop_raw`. Skipping."
elif [[ ! -e $SCENE_PATH/bop_raw/scene_gt_initial.json ]] || \
    [[ ! -e $SCENE_PATH/bop_raw/scene_camera_initial.json ]]
then
    error "Unable to find required input files in `path $SCENE_PATH/bop_raw`. Exiting."
elif [[ ! -e $SCENE_PATH/canonical_pose.json ]]
then
    error "Unable to find canonical pose `path $SCENE_PATH/canonical_pose.json`. Exiting."
elif [[ ! -e $SCENE_PATH/canonical_scale.json ]]
then
    error "Unable to find canonical scale `path $SCENE_PATH/canonical_scale.json`. Exiting."
else
    info "Applying canonical pose and scale to BOP annotations..."
    python $SOURCE_PATH/scripts/apply_canonical_pose_and_scale.py \
        --canonical_pose_fn $SCENE_PATH/canonical_pose.json \
        --canonical_scale_fn $SCENE_PATH/canonical_scale.json \
        --input_bop_gt_fn $SCENE_PATH/bop_raw/scene_gt_initial.json \
        --input_bop_camera_fn $SCENE_PATH/bop_raw/scene_camera_initial.json \
        --output_bop_gt_fn $SCENE_PATH/bop_raw/scene_gt_canonical.json \
        --output_bop_camera_fn $SCENE_PATH/bop_raw/scene_camera_canonical.json ||
    exit_with_error
fi


##################################################
# Prompt for alignment to reference
##################################################
if [[ $CONFIG_IS_REFERENCE == 1 ]]
then
    header "Finalizing reference scene"
    info "Found config flag `path '$CONFIG_IS_REFERENCE == 1'`."
    if [[ -e $SCENE_PATH/bop_raw/scene_gt.json ]] &&
        [[ -e $SCENE_PATH/bop_raw/scene_camera.json ]] &&
        [[ -e $SCENE_PATH/meshes/reference.ply ]]
    then 
        info "Found existing reference annotations and mesh. Skipping."
    elif [[ ! -e $SCENE_PATH/bop_raw/scene_gt_canonical.json ]] || \
        [[ ! -e $SCENE_PATH/bop_raw/scene_camera_canonical.json ]]
    then
        error "Unable to find required input files in `path $SCENE_PATH/bop_raw`. Exiting."
    elif [[ ! -e $SCENE_PATH/meshes/scaled.ply ]]
    then
        error "Unable to find scaled mesh `path $SCENE_PATH/meshes/scaled.ply`. Exiting."
    else
        info "Copying canonical annotations to final annotations and scaled mesh to reference mesh ..."
        cp $SCENE_PATH/bop_raw/scene_gt_canonical.json $SCENE_PATH/bop_raw/scene_gt.json &&
        cp $SCENE_PATH/bop_raw/scene_camera_canonical.json $SCENE_PATH/bop_raw/scene_camera.json &&
        cp $SCENE_PATH/meshes/scaled.ply $SCENE_PATH/meshes/reference.ply ||
        exit_with_error
    fi
else
    header "Aligning to reference mesh"
    info "Found config flag `path '$CONFIG_IS_REFERENCE != 1'`."
    if [[ -e $SCENE_PATH/reference_alignment_pose.json ]] && \
        [[ -e $SCENE_PATH/reference_alignment_scale.json ]] && \
        [[ -e $SCENE_PATH/meshes/aligned_to_reference.ply ]]
    then 
        info "Found existing `path $SCENE_PATH/reference_alignment_*.json`. Skipping."
    elif [[ $IS_INTERACTIVE != 1 ]] 
    then 
        info "Unable to find existing `path $SCENE_PATH/reference_alignment_*.json`."
        error "Aligning to reference requires interactive GUI tool. Use flag `path --interactive` to run. Exiting."
    elif [[ ! -e $SCENE_PATH/meshes/scaled.ply ]]
    then 
        error "Unable to find canonical scaled mesh `path $SCENE_PATH/meshes/scaled.ply`. Exiting."
    elif [[ $CONFIG_REFERENCE_SCENE == "" ]]
    then
        error "CONFIG_REFERENCE_SCENE is not set. Exiting."
    elif [[ ! -e $SCENE_PATH/$CONFIG_REFERENCE_SCENE/meshes/reference.ply ]]
    then 
        error "Unable to find reference mesh `path $SCENE_PATH/$CONFIG_REFERENCE_SCENE/meshes/reference.ply`. Exiting."
    else
        info "Running interactive tool to align to reference mesh ..."
        python $SOURCE_PATH/scripts/manual_align.py \
            $SCENE_PATH/meshes/scaled.ply \
            --already_in_canonical_pose \
            --output_transform_fn $SCENE_PATH/reference_alignment_pose.json \
            --output_scale_fn $SCENE_PATH/reference_alignment_scale.json \
            --output_mesh_fn $SCENE_PATH/meshes/aligned_to_reference.ply \
            --reference_mesh $SCENE_PATH/$CONFIG_REFERENCE_SCENE/meshes/reference.ply  \
            --allow_translation \
            --allow_scaling &&
        cp $SCENE_PATH/$CONFIG_REFERENCE_SCENE/meshes/reference.ply $SCENE_PATH/meshes/reference.ply ||
        exit_with_error
    fi

    header "Applying reference pose alignment"
    if [[ -e $SCENE_PATH/bop_raw/scene_gt.json ]] && \
        [[ -e $SCENE_PATH/bop_raw/scene_camera.json ]]
    then 
        info "Found existing output files in `path $SCENE_PATH/bop_raw`. Skipping."
    elif [[ ! -e $SCENE_PATH/bop_raw/scene_gt_canonical.json ]] || \
        [[ ! -e $SCENE_PATH/bop_raw/scene_camera_canonical.json ]]
    then
        error "Unable to find required input files in `path $SCENE_PATH/bop_raw`. Exiting."
    elif [[ ! -e $SCENE_PATH/canonical_pose.json ]]
    then
        error "Unable to find canonical pose `path $SCENE_PATH/canonical_pose.json`. Exiting."
    elif [[ ! -e $SCENE_PATH/canonical_scale.json ]]
    then
        error "Unable to find canonical scale `path $SCENE_PATH/canonical_scale.json`. Exiting."
    else
        info "Applying canonical pose and scale to BOP annotations..."
        python $SOURCE_PATH/scripts/apply_canonical_pose_and_scale.py \
            --canonical_pose_fn $SCENE_PATH/reference_alignment_pose.json \
            --canonical_scale_fn $SCENE_PATH/reference_alignment_scale.json \
            --input_bop_gt_fn $SCENE_PATH/bop_raw/scene_gt_canonical.json \
            --input_bop_camera_fn $SCENE_PATH/bop_raw/scene_camera_canonical.json \
            --output_bop_gt_fn $SCENE_PATH/bop_raw/scene_gt.json \
            --output_bop_camera_fn $SCENE_PATH/bop_raw/scene_camera.json||
        exit_with_error
    fi
fi


##################################################
# Generate BOP masks
##################################################
header "Generating BOP masks"
mask_files=($SCENE_PATH/bop_raw/mask/*.png)
mask_visib_files=($SCENE_PATH/bop_raw/mask_visib/*.png)
if [[ -e $mask_files ]] && [[ -e $mask_visib_files ]]
then 
    info "Found existing BOP masks in `path $SCENE_PATH/bop_raw/mask` and `path $SCENE_PATH/bop_raw/mask_visib`. Skipping."
elif
    [[ ! -e $SCENE_PATH/meshes/reference.ply ]] || \
    [[ ! -e $SCENE_PATH/bop_raw/scene_gt.json ]] || \
    [[ ! -e $SCENE_PATH/bop_raw/scene_camera.json ]] || \
    [[ ! -e $SCENE_PATH/depth_scene/ ]]
then
    error "Unable to find required input files for mask generation. Exiting."
else
    info "Generating BOP masks in `path $SCENE_PATH/bop_raw` ..."
    python $SOURCE_PATH/scripts/generate_bop_masks.py \
        $SCENE_PATH/bop_raw \
        --mask_types visible full \
        --scene_depth_dir $SCENE_PATH/depth_scene \
        --scene_gt_fn $SCENE_PATH/bop_raw/scene_gt_canonical.json \
        --scene_camera_fn $SCENE_PATH/bop_raw/scene_camera_canonical.json \
        --scene_mesh_fn $SCENE_PATH/meshes/scaled.ply \
        --visible_tolerance $CONFIG_VISIBILITY_TOLERANCE ||
    exit_with_error
fi


##################################################
# Assemble BOP annotations
##################################################
header "Assembling final BOP annotations"
info "Copying final BOP annotations to `path $SCENE_PATH/bop` ..."
if [[ -e $SCENE_PATH/bop/rgb ]] && \
    [[ -e $SCENE_PATH/bop/depth_nerf ]] && \
    [[ -e $SCENE_PATH/bop/mask ]] && \
    [[ -e $SCENE_PATH/bop/mask_visib ]] && \
    [[ -e $SCENE_PATH/bop/scene_gt.json ]] && \
    [[ -e $SCENE_PATH/bop/scene_camera.json ]] && \
    [[ -e $SCENE_PATH/bop/obj_$(printf "%06d" $CONFIG_BOP_ID).ply ]]
then 
    info "Found existing BOP annotations in `path $SCENE_PATH/bop`. Skipping."
elif [[ ! -e $SCENE_PATH/images_scene/ ]] || \
    [[ ! -e $SCENE_PATH/depth_scene/ ]] || \
    [[ ! -e $SCENE_PATH/bop_raw/mask/ ]] || \
    [[ ! -e $SCENE_PATH/bop_raw/mask_visib/ ]] || \
    [[ ! -e $SCENE_PATH/bop_raw/scene_gt.json ]] || \
    [[ ! -e $SCENE_PATH/bop_raw/scene_camera.json ]] || \
    [[ ! -e $SCENE_PATH/meshes/reference.ply ]]
then
    error "Unable to find required input files for final BOP annotations. Exiting."
else
    mkdir -p $SCENE_PATH/bop &&
    cp -r $SCENE_PATH/images_scene $SCENE_PATH/bop/rgb &&
    cp -r $SCENE_PATH/depth_scene $SCENE_PATH/bop/depth_nerf &&
    cp -r $SCENE_PATH/bop_raw/mask $SCENE_PATH/bop/mask &&
    cp -r $SCENE_PATH/bop_raw/mask_visib $SCENE_PATH/bop/mask_visib &&
    cp $SCENE_PATH/bop_raw/scene_gt.json $SCENE_PATH/bop/scene_gt.json &&
    cp $SCENE_PATH/bop_raw/scene_camera.json $SCENE_PATH/bop/scene_camera.json &&
    cp $SCENE_PATH/meshes/reference.ply $SCENE_PATH/bop/obj_$(printf "%06d" $CONFIG_BOP_ID).ply ||
    exit_with_error
fi


##################################################
# Done!
##################################################
header "Done!"
info "Final annotations are written to `path $SCENE_PATH/bop`."
exit 0
