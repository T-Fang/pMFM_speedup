function generate_inputs_for_split(split_name)
    %GENERATE_INPUTS_FOR_SPLIT generate inputs for pMFM for a given split (either 'train', 'validation' or 'test')
    subject_groups = load_subject_groups(['/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/' split_name '_groups.csv']);

    for i = 1:length(subject_groups)
        subject_group = subject_groups(i, :);
        group_dir = ['/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/' split_name '/' num2str(i)];

        if ~exist(group_dir, 'dir')
            mkdir(group_dir)
        end

        group_level_SC = generate_group_level_SC(subject_group);
        csvwrite([group_dir '/group_level_SC.csv'], group_level_SC)
        [group_level_FC, group_level_FCD] = generate_group_level_FC_FCD(subject_group);
        csvwrite([group_dir '/group_level_FC.csv'], group_level_FC)
        csvwrite([group_dir '/group_level_FCD.csv'], group_level_FCD)
        save([group_dir '/group_level_FCD.mat'], 'group_level_FCD')
        [group_level_myelin, ~, ~, ~] = generate_group_level_myelin(subject_group);
        csvwrite([group_dir '/group_level_myelin.csv'], group_level_myelin)
        [group_level_RSFC_gradient, ~, ~, ~] = generate_group_level_RSFC_gradient(subject_group);
        csvwrite([group_dir '/group_level_RSFC_gradient.csv'], group_level_RSFC_gradient)
    end

end
