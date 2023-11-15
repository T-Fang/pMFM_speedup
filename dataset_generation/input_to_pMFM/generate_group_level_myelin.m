function [group_level_myelin_ave, group_level_myelin_pca, var_explain, subject_count] = generate_group_level_myelin(subject_list)
    % This function is used to generate group-level myelin by averaging myelin
    % across subjects

    % subject count is the numebr of subjects with myelin map
    % group_level_myelin_pca are the PCs (7? 52% var explained)
    % var_explain is the variance explained by these PCs

    % deiskan parcellation
    lh_desikan_fslr_32k = CBIG_read_fslr_surface('lh', 'fs_LR_32k', 'very_inflated', 'aparc.annot');
    lh_desikan_fslr_32k_label = lh_desikan_fslr_32k.MARS_label;
    rh_desikan_fslr_32k = CBIG_read_fslr_surface('rh', 'fs_LR_32k', 'very_inflated', 'aparc.annot');
    rh_desikan_fslr_32k_label = rh_desikan_fslr_32k.MARS_label + 36;
    combined_desikan_label = [lh_desikan_fslr_32k_label; rh_desikan_fslr_32k_label];

    myelin_all = [];
    subject_count = 0;

    for i = 1:length(subject_list)
        disp(i)
        sub_id = subject_list(i);

        if isa(sub_id, 'double')
            sub_id = num2str(sub_id);
        end

        myelin_map_path = ['/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/' ...
                            'individuals/' sub_id '/MNINonLinear/' ...
                            'fsaverage_LR32k/' sub_id '.SmoothedMyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'];

        if exist(myelin_map_path, 'file')
            subject_count = subject_count + 1;
            myelin_map_cifti = ft_read_cifti(myelin_map_path);
            myelin_map = myelin_map_cifti.smoothedmyelinmap_bc_msmall;

            for j = 1:64984

                if combined_desikan_label(j) == 1 || ...
                        combined_desikan_label(j) == 5 || ...
                        combined_desikan_label(j) == 37 || ...
                        combined_desikan_label(j) == 41
                    myelin_map(j) = 0;
                end

            end

            % average RSFC gradient within each ROI
            downsampled_desikan_subject_myelin_map = zeros(72, 1);

            for roi = 1:72
                ROI_mask = (combined_desikan_label == roi);
                ROI_value = nanmean(myelin_map(ROI_mask));
                downsampled_desikan_subject_myelin_map(roi) = ROI_value;
            end

            % remove medial wall and corpus callosum
            downsampled_desikan_subject_myelin_map(41) = [];
            downsampled_desikan_subject_myelin_map(37) = [];
            downsampled_desikan_subject_myelin_map(5) = [];
            downsampled_desikan_subject_myelin_map(1) = [];
            desikan_myelin_map = downsampled_desikan_subject_myelin_map;

            myelin_all = [myelin_all desikan_myelin_map]; % 68 x num_subject
        end

    end

    [coeff, ~, ~, ~, explained, group_level_myelin_ave] = pca(myelin_all');
    group_level_myelin_pca = coeff(:, 1:10);
    var_explain = explained(1:10);
    group_level_myelin_ave = group_level_myelin_ave';

end
