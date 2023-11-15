function [group_FC, group_FCD_CDF] = generate_group_level_FC_FCD(subject_list)

    % This is used to generate group FC and FCD CDF for group-level training
    % leaving HCP retest group out (during the training, validation and test)

    roi_list = ones(68, 1);
    FC_all = [];
    FCD_CDF_all = [];
    num_roi = sum(roi_list);

    for i = 1:length(subject_list)
        subject = num2str(subject_list(i));

        path_1 = ['/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/pMFM_TC/all/' subject '_rfMRI_REST1_LR.mat'];

        if exist(path_1, 'file')
            TC = load(path_1);
            TC = TC.TC;
            TC([1, 5, 37, 41], :) = [];
            TC = TC .* roi_list;
            TC = TC(any(TC, 2), :);

            if size(TC, 2) == 1200
                TC = TC';
                FC = CBIG_self_corr(TC);
                FC_all = cat(3, FC_all, FC);

                FCD_run = zeros(num_roi * (num_roi - 1) / 2, 1200 - 82);

                for j = 1:(1200 - 82)
                    TC_section = TC(j:j + 82, :); % size 83x68
                    FC_section = CBIG_self_corr(TC_section); % size 68x68
                    FC_vec_section = FC_section(triu(true(size(FC_section, 1)), 1)); % size 2278x1
                    FCD_run(:, j) = FC_vec_section;
                end

                FCD_run = corr(FCD_run); % size (1200-82) x (1200-82)
                FCD_run_vec = FCD_run(triu(true(size(FCD_run, 1)), 1)); % size 624403x1
                bin_count = histcounts(sort(FCD_run_vec), -1:0.0002:1); % 10000 bins
                FCD_CDF = cumsum(bin_count); % cumulative sum size 1x10000
                FCD_CDF_all = cat(1, FCD_CDF_all, FCD_CDF);
            end

        end

        path_2 = ['/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/pMFM_TC/all/' subject '_rfMRI_REST1_RL.mat'];

        if exist(path_2, 'file')
            TC = load(path_2);
            TC = TC.TC;
            TC([1, 5, 37, 41], :) = [];
            TC = TC .* roi_list;
            TC = TC(any(TC, 2), :);

            if size(TC, 2) == 1200
                TC = TC';
                FC = CBIG_self_corr(TC);
                FC_all = cat(3, FC_all, FC);

                FCD_run = zeros(num_roi * (num_roi - 1) / 2, 1200 - 82);

                for j = 1:(1200 - 82)
                    TC_section = TC(j:j + 82, :); % size 83x68
                    FC_section = CBIG_self_corr(TC_section); % size 68x68
                    FC_vec_section = FC_section(triu(true(size(FC_section, 1)), 1)); % size 2278x1
                    FCD_run(:, j) = FC_vec_section;
                end

                FCD_run = corr(FCD_run); % size (1200-82) x (1200-82)
                FCD_run_vec = FCD_run(triu(true(size(FCD_run, 1)), 1)); % size 624403x1
                bin_count = histcounts(sort(FCD_run_vec), -1:0.0002:1); % 10000 bins
                FCD_CDF = cumsum(bin_count); % cumulative sum size 1x10000
                FCD_CDF_all = cat(1, FCD_CDF_all, FCD_CDF);
            end

        end

        path_3 = ['/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/pMFM_TC/all/' subject '_rfMRI_REST2_LR.mat'];

        if exist(path_3, 'file')
            TC = load(path_3);
            TC = TC.TC;
            TC([1, 5, 37, 41], :) = [];
            TC = TC .* roi_list;
            TC = TC(any(TC, 2), :);

            if size(TC, 2) == 1200
                TC = TC';
                FC = CBIG_self_corr(TC);
                FC_all = cat(3, FC_all, FC);

                FCD_run = zeros(num_roi * (num_roi - 1) / 2, 1200 - 82);

                for j = 1:(1200 - 82)
                    TC_section = TC(j:j + 82, :); % size 83x68
                    FC_section = CBIG_self_corr(TC_section); % size 68x68
                    FC_vec_section = FC_section(triu(true(size(FC_section, 1)), 1)); % size 2278x1
                    FCD_run(:, j) = FC_vec_section;
                end

                FCD_run = corr(FCD_run); % size (1200-82) x (1200-82)
                FCD_run_vec = FCD_run(triu(true(size(FCD_run, 1)), 1)); % size 624403x1
                bin_count = histcounts(sort(FCD_run_vec), -1:0.0002:1); % 10000 bins
                FCD_CDF = cumsum(bin_count); % cumulative sum size 1x10000
                FCD_CDF_all = cat(1, FCD_CDF_all, FCD_CDF);
            end

        end

        path_4 = ['/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/pMFM_TC/all/' subject '_rfMRI_REST2_RL.mat'];

        if exist(path_4, 'file')
            TC = load(path_4);
            TC = TC.TC;
            TC([1, 5, 37, 41], :) = [];
            TC = TC .* roi_list;
            TC = TC(any(TC, 2), :);

            if size(TC, 2) == 1200
                TC = TC';
                FC = CBIG_self_corr(TC);
                FC_all = cat(3, FC_all, FC);

                FCD_run = zeros(num_roi * (num_roi - 1) / 2, 1200 - 82);

                for j = 1:(1200 - 82)
                    TC_section = TC(j:j + 82, :); % size 83x68
                    FC_section = CBIG_self_corr(TC_section); % size 68x68
                    FC_vec_section = FC_section(triu(true(size(FC_section, 1)), 1)); % size 2278x1
                    FCD_run(:, j) = FC_vec_section;
                end

                FCD_run = corr(FCD_run); % size (1200-82) x (1200-82)
                FCD_run_vec = FCD_run(triu(true(size(FCD_run, 1)), 1)); % size 624403x1
                bin_count = histcounts(sort(FCD_run_vec), -1:0.0002:1); % 10000 bins
                FCD_CDF = cumsum(bin_count); % cumulative sum size 1x10000
                FCD_CDF_all = cat(1, FCD_CDF_all, FCD_CDF);
            end

        end

    end

    FC_sum = zeros(size(FC, 1));

    for i = 1:size(FC_all, 3)
        FC = FC_all(:, :, i);
        FC_sum = FC_sum + CBIG_StableAtanh(FC);
    end

    group_FC = tanh(FC_sum / size(FC_all, 3));
    group_FCD_CDF = mean(FCD_CDF_all);
    group_FCD_CDF = round(group_FCD_CDF');

end
