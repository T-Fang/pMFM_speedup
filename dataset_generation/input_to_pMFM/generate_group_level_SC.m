function group_level_SC = generate_group_level_SC(subject_list)
    % subject list is HCP subject from resting state !!!

    roi_list = ones(72, 1);
    roi_list([1, 5, 37, 41]) = [];
    num_roi = sum(roi_list, 1);

    group_level_SC = zeros(68);
    SC_subject_compilation = zeros(68, 68, length(subject_list));

    for i = 1:length(subject_list)
        subject = subject_list(i);
        SC_subject = csvread(['/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_S1200_diffusion_preprocessed/tractography/iFOD2/No_subcortical_Leon/' num2str(subject) '/connectomes/connectome_DK_82Parcels_SIFT2.csv']);
        SC_subject(35:48, :) = []; % exlcuding subcortical /home/shaoshi.z/storage/MRtrix/pMFM/script/fs_DK_82.txt
        SC_subject(:, 35:48) = [];
        SC_subject(SC_subject < 0) = 0;
        SC_subject_compilation(:, :, i) = SC_subject;
    end

    % keep track of how many subjects have zero in that entry
    zero_count_map = zeros(68, 68);

    for i = 1:68

        for j = 1:68
            SC_entry_all = SC_subject_compilation(i, j, :);
            SC_entry_all = SC_entry_all(:);
            zero_count_map(i, j) = sum(SC_entry_all == 0);
        end

    end

    SC_mask = zero_count_map <= length(subject_list) / 2;

    for i = 1:length(subject_list)
        SC_subject = SC_subject_compilation(:, :, i);
        SC_subject_mask = SC_mask .* SC_subject;
        SC_subject_compilation(:, :, i) = SC_subject_mask;
    end

    for i = 1:68

        for j = 1:68
            SC_entry_all = SC_subject_compilation(i, j, :);
            SC_entry_all = SC_entry_all(:);
            SC_entry_all_log = log(SC_entry_all);
            SC_entry_all_log(isinf(SC_entry_all_log)) = 0;
            group_level_SC(i, j) = mean(nonzeros(SC_entry_all_log));
        end

    end

    group_level_SC(isnan(group_level_SC)) = 0;
    group_level_SC(logical(eye(68))) = 0; % comment this out if we want diagonal SC entries

    roi_list = (1:68) .* roi_list';
    roi_list(roi_list == 0) = [];
    roi_list = setdiff(1:68, roi_list); %roi remove list

    group_level_SC(roi_list, :) = [];
    group_level_SC(:, roi_list) = [];
end
