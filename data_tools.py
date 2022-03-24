import json


def read_json_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def split_data(data):
    """
    划分训练集、验证集和测试集
    :param data:
    :return:
    """
    t_path = "./data/id_ans_test"
    v_path = "./data/valid_ids.json"
    valid_id = read_json_data(v_path)
    test_id = []
    with open(t_path, 'r') as f:
        for line in f:
            test_id.append(line.strip().split('\t')[0])
    train, test, valid = [], [], []
    for key, value in data.items():
        if key in test_id:
            test.append((key, value))
        elif key in valid_id:
            valid.append((key, value))
        else:
            train.append((key, value))
    return train, valid, test


def string_2_idx_sen(sen,  vocab_dict):
    """
    返回句子的onehot
    """
    return [vocab_dict[word] for word in sen]


def pad_sen(sen_idx_list, max_len=115, pad_idx=0):
    return sen_idx_list + [pad_idx] * (max_len - len(sen_idx_list))


def post_solver(post_equ):
    """
    计算后缀表达式的值
    :param post_equ:
    :return:
    """
    stack = []
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        if elem not in op_list:
            op_v = elem
            if '%' in op_v:
                op_v = float(op_v[:-1])/100.0
            stack.append(str(op_v))
        elif elem in op_list:
            op_v_1 = stack.pop()
            op_v_1 = float(op_v_1)
            op_v_2 = stack.pop()
            op_v_2 = float(op_v_2)
            if elem == '+':
                stack.append(str(op_v_2+op_v_1))
            elif elem == '-':
                stack.append(str(op_v_2-op_v_1))
            elif elem == '*':
                stack.append(str(op_v_2*op_v_1))
            elif elem == '/':
                stack.append(str(op_v_2/op_v_1))
            else:
                stack.append(str(op_v_2**op_v_1))
    return stack.pop()


def inverse_temp_to_num(equ_list, num_list):
    """
    将表达式中的temp转换回数字
    :param equ_list:
    :param num_list:
    :return:
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new_equ_list = []
    for elem in equ_list:
        if 'temp' in elem:
            index = alphabet.index(elem[-1])
            try:
                new_equ_list.append(num_list[index])
            except:
                return []
        elif 'PI' == elem:
            new_equ_list.append('3.14')
        else:
            new_equ_list.append(elem)
    return new_equ_list