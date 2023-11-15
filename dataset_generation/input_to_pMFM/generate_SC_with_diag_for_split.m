function generate_SC_with_diag_for_split(split_name)
    %GENERATE_INPUTS_FOR_SPLIT generate inputs for pMFM for a given split (either 'train', 'validation' or 'test')
    subject_groups = load_subject_groups(['/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/' split_name '_groups.csv']);

    for i = 1:length(subject_groups)
        subject_group = subject_groups(i, :);
        group_dir = ['/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/' split_name '/' num2str(i)];

        if ~exist(group_dir, 'dir')
            mkdir(group_dir)
        end

        group_level_SC_with_diag = generate_group_level_SC_with_diag(subject_group);
        csvwrite([group_dir '/group_level_SC_with_diag.csv'], group_level_SC_with_diag)
    end

end
