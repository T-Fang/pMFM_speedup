function [group_level_RSFC_gradient_ave, group_level_RSFC_gradient_pca, var_explain, subject_count] = generate_group_level_RSFC_gradient(subject_list)
    % This function is used to generate group-level RSFC gradient by averaging
    % the principal gradient across subjects

    % subject_count is the number of subejcts with RSFC gradient
    % frooup_level_RSFC_gradient_pca is the top PCs
    % var_explain is varaince explained by these PCs

    lh_desikan_fslr_32k = CBIG_read_fslr_surface('lh', 'fs_LR_32k', 'very_inflated', 'aparc.annot');
    lh_desikan_fslr_32k_label = lh_desikan_fslr_32k.MARS_label;
    rh_desikan_fslr_32k = CBIG_read_fslr_surface('rh', 'fs_LR_32k', 'very_inflated', 'aparc.annot');
    rh_desikan_fslr_32k_label = rh_desikan_fslr_32k.MARS_label + 36;
    combined_desikan_label = [lh_desikan_fslr_32k_label; rh_desikan_fslr_32k_label];

    lh_avg_mesh = CBIG_read_fslr_surface('lh', 'fs_LR_32k', 'sphere', 'medialwall.annot');
    lh_avg_mesh_label = lh_avg_mesh.MARS_label;
    rh_avg_mesh = CBIG_read_fslr_surface('rh', 'fs_LR_32k', 'sphere', 'medialwall.annot');
    rh_avg_mesh_label = rh_avg_mesh.MARS_label;
    combined_medial_wall_label = [lh_avg_mesh_label; rh_avg_mesh_label];
    combined_medial_wall_label = combined_medial_wall_label == 1;

    RSFC_gradient_all = [];
    subject_count = 0;

    for i = 1:length(subject_list)
        sub_id = subject_list(i);
        % disp(i)

        if isa(sub_id, 'double')
            sub_id = num2str(sub_id);
        end

        subject_gradient_path = ['/mnt/nas/CSC7/Yeolab/PreviousLabMembers/tnyr/Diffusion_Embedding/FC_matrices/' ...
                                '746_Group_FC/Aligned_Gradient/' sub_id '/' sub_id '_realigned_mat.mat'];
        % disp(subject_gradient_path)

        if exist(subject_gradient_path, 'file')
            subject_count = subject_count + 1;
            subject_gradient_no_medial_wall = load(subject_gradient_path);
            % only the first principal RSFC gradient
            subject_gradient_no_medial_wall = subject_gradient_no_medial_wall.realigned_mat(:, 1);

            subject_gradient = zeros(64984, 1); %initiliazaion
            count = 1;
            % fill in the medial wall vertices with 0
            for j = 1:64984

                if combined_medial_wall_label(j) == 0
                    subject_gradient(j) = subject_gradient_no_medial_wall(count);
                    count = count + 1;
                end

            end

            % average RSFC gradient within each ROI
            downsampled_desikan_subject_gradient = zeros(72, 1);

            for roi = 1:72
                ROI_mask = (combined_desikan_label == roi);
                ROI_value = nanmean(subject_gradient(ROI_mask));

                if isnan(ROI_value)
                    ROI_value = 0;
                end

                downsampled_desikan_subject_gradient(roi) = ROI_value;
            end

            % remove medial wall and corpus callosum
            downsampled_desikan_subject_gradient(41) = [];
            downsampled_desikan_subject_gradient(37) = [];
            downsampled_desikan_subject_gradient(5) = [];
            downsampled_desikan_subject_gradient(1) = [];

            RSFC_gradient_all = [RSFC_gradient_all downsampled_desikan_subject_gradient];
        end

    end

    [coeff, ~, ~, ~, explained, group_level_RSFC_gradient_ave] = pca(RSFC_gradient_all');
    group_level_RSFC_gradient_pca = coeff(:, 1:10);
    var_explain = explained(1:10);
    group_level_RSFC_gradient_ave = -1 * group_level_RSFC_gradient_ave';
end
