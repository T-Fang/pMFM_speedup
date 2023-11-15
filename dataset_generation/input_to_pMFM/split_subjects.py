from random import shuffle
import csv


def split_subjects(path_to_all_subject_list):
    with open(
            path_to_all_subject_list, 'r', encoding='utf-8',
            errors='ignore') as f:
        subjects = f.read().split('\n')
    assert len(subjects) == 1000
    shuffle(subjects)
    subjects_train = subjects[:610]
    subjects_validation = subjects[610:790]
    subjects_test = subjects[790:]

    train_groups = []
    for i in range(57):
        train_groups.append(subjects_train[i * 10:i * 10 + 50])

    validation_groups = []
    for i in range(14):
        validation_groups.append(subjects_validation[i * 10:i * 10 + 50])

    test_groups = []
    for i in range(17):
        test_groups.append(subjects_test[i * 10:i * 10 + 50])

    with open(
            './train_groups.csv', 'w+', encoding='utf-8',
            errors='ignore') as f:
        csv_writer = csv.writer(f)
        for group in train_groups:
            csv_writer.writerow(group)

    with open(
            './validation_groups.csv', 'w+', encoding='utf-8',
            errors='ignore') as f:
        csv_writer = csv.writer(f)
        for group in validation_groups:
            csv_writer.writerow(group)

    with open(
            './test_groups.csv', 'w+', encoding='utf-8', errors='ignore') as f:
        csv_writer = csv.writer(f)
        for group in test_groups:
            csv_writer.writerow(group)


if __name__ == '__main__':
    # There's no need to re-run this script if the subject grouping is already generated
    path_to_all_subject_list = './all_subject_list.txt'
    split_subjects(path_to_all_subject_list)
    pass
