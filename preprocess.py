import csv
import pubchempy
import numpy as np
import pandas as pd
import math
from functools import reduce

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""


def load_drug_list():
    drug_list = pd.read_csv('./data/drug_list.csv')
    drugs = drug_list['drug_name'].to_list()
    drugs = list(set(drugs))
    return drugs


def write_drug_cid():
    drugs = load_drug_list()
    unknow_drugs = []
    with open('./data/pychem_cid.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['drug_name', 'CID'])
        for drug in drugs:
            c = pubchempy.get_compounds(drug, 'name')

            if len(c) != 0:
                cid = c[0].cid
                row = [drug, str(cid)]
                print(row)
                wr.writerow(row)

            else:
                unknow_drugs.append(drug)
                print('{} is not found by pychem'.format(drug))

    with open('./data/unknow_drug_by_pychem.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(unknow_drugs)


# def cid_from_other_source():
#     """
#     some drug can not be found in pychem, so I try to find some cid manually.
#     the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
#     """
#     f = open(folder + "small_molecule.csv", 'r')
#     reader = csv.reader(f)
#     next(reader)
#     cid_dict = {}
#     for item in reader:
#         name = item[1]
#         cid = item[4]
#         if not name in cid_dict:
#             cid_dict[name] = str(cid)
#
#     unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
#     drug_cid_dict = {k: v for k, v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
#     return drug_cid_dict


# def load_cid_dict():
#     """
#     pubchem中找不到的药，还需要进一步研究如何搜索SMILES
#     :return:
#     """
#     reader = csv.reader(open(folder + "pychem_cid.csv"))
#     pychem_dict = {}
#     for item in reader:
#         pychem_dict[item[0]] = item[1]
#     # pychem_dict.update(cid_from_other_source())
#     return pychem_dict


def download_smiles():
    pychem_cid = pd.read_csv('./data/pychem_cid.csv')
    pubchempy.download('CSV', './data/drug_smiles.csv', pychem_cid['CID'].to_list(),
                       operation='property/CanonicalSMILES,IsomericSMILES',
                       overwrite=True)
    drug_smiles = pd.read_csv('./data/drug_smiles.csv')
    pychem_cid = pychem_cid.merge(drug_smiles, on='CID')
    pychem_cid.to_csv("./data/drug_smiles.csv", index=False)


"""
The following code will convert the SMILES format into onehot format
"""


def string2smiles_list(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i - 1:i + 1])
            i += 1
        else:
            char_list.append(string[i - 1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list


def onehot_encode(char_list, smiles_string, length):
    encode_row = lambda char: map(int, [c == char for c in smiles_string])
    ans = list(map(encode_row, char_list))
    ans = [list(x) for x in ans]
    ans = np.array(ans)
    if ans.shape[1] < length:
        residual = np.zeros((len(char_list), length - ans.shape[1]), dtype=np.int8)
        ans = np.concatenate((ans, residual), axis=1)
    return ans


def charsets(smiles):
    union = lambda x, y: set(x) | set(y)
    c_chars = list(reduce(union, map(string2smiles_list, list(smiles['CanonicalSMILES']))))
    c_chars.sort()
    c_length = max(map(len, map(string2smiles_list, list(smiles['CanonicalSMILES']))))
    return c_chars, c_length


def save_drug_smiles_onehot():
    smiles = pd.read_csv('./data/drug_smiles.csv')
    c_chars, c_length = charsets(smiles)
    print(c_chars)
    print('max length:{}'.format(c_length))
    drug_dict = {}
    for i in range(len(smiles)):
        drug_dict[smiles.iloc[i, 0]] = onehot_encode(c_chars, smiles.iloc[i, 2], c_length)
    np.save('./data/drug_feature.npy', drug_dict)
    return drug_dict


"""
The following part will prepare the mutation features for the cell.
"""


def save_cell_mut_matrix():
    f = open("./data/PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    cell_dict = {}
    mut_dict = {}

    matrix_list = []
    organ1_dict = {}
    organ2_dict = {}
    for item in reader:
        cell = item[0]
        mut = item[5]
        organ1_dict[cell] = item[2]
        organ2_dict[cell] = item[3]
        is_mutated = int(item[6])
        if cell in cell_dict:
            row = cell_dict[cell]
        else:
            row = len(cell_dict)
            cell_dict[cell] = row
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col
        matrix_list.append((row, col, is_mutated))

    matrix = np.ones(shape=(len(cell_dict), len(mut_dict)), dtype=np.float32)
    matrix = matrix * -1
    for item in matrix_list:
        matrix[item[0], item[1]] = item[2]

    feature_num = [len(list(filter(lambda x: x >= 0, list(matrix[i, :])))) for i in range(len(cell_dict))]
    indics = [i for i in range(len(feature_num)) if feature_num[i] == 735]
    matrix = matrix[indics, :]

    inv_cell_dict = {v: k for k, v in cell_dict.items()}
    all_names = [inv_cell_dict[i] for i in range(len(inv_cell_dict))]
    cell_names = np.array([all_names[i] for i in indics])

    cell_dict = {}
    for i in range(len(matrix)):
        cell_dict[cell_names[i]] = matrix[i]

    np.save("./data/cell_feature.npy", cell_dict)
    print("finish saving cell mut data!")

    return cell_dict


def main():
    write_drug_cid()
    download_smiles()
    save_drug_smiles_onehot()
    save_cell_mut_matrix()


if __name__ == "__main__":
    main()
