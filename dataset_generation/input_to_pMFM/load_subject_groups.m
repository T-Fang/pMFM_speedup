function subject_lists = load_subject_groups(group_list_path)
    table = readtable(group_list_path);
    subject_lists = table2array(table);
end
