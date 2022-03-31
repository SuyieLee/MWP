import json
import jieba
import random
random.seed(10)

def get_math23k_data(dataname):
    """
    读取文件，按行读取
    :param dataname:
    :return:
    """
    with open(dataname, 'r') as f:
        data_list = []
        st = ""
        count = 0
        for line in f:
            count+=1
            st += line
            if count % 7 == 0:
                data_list.append(json.loads(st))
                st = ""
    return data_list


def get_json_data(dataname):
    with open(dataname, 'r') as f:
        return json.load(f)


def mask_number(split_text, equation):
    """"
    把数字替换成特定的字母，temp_a,temp_b...
    文本里要换，提取出数字表，公式里面也换
    """
    alphas = 'abcdefghijklmnopqrstuvwxyz'
    text = []
    number_list = []
    count = 0
    equation_list = []
    for word in split_text:
        if word[0] in '0123456789':
            text.append('temp_'+alphas[count])
            if '%' in word:
                text.append('%')
            count += 1
            number_list.append(word)
        elif word[0] == '(' and len(word) > 1:
            text.append('temp_'+alphas[count])
            count += 1
            number_list.append(word)
        # elif '[' in word:
        #     print(split_text)
        #     print(equation)
        #     print(word)
        else:
            text.append(word)

    s_n = sorted([(w, i) for i, w in enumerate(number_list)], key=lambda x: len(str(x[0])), reverse=True)

    for num, idx in s_n:
        equation = equation.replace(num,'temp_'+alphas[idx], 15)

    flag = 0
    step = ""
    for ch in equation:
        if flag == 6:
            equation_list.append(step)
            flag = 0
            step = ""

        if ch == 't' or flag > 0:
            step += ch
            flag += 1
        else:
            equation_list.append(ch)
            flag = 0
    if flag > 0:
        equation_list.append(step)
    return text, number_list, equation_list


def split_num_unit(numU):
    """
    将数字和单位分离开，例如80km，191KM这种
    :param numU:
    :return:
    """
    numU = numU.lower()
    alphas = 'abcdefghijklmnopqrstuvwxyz'
    st = ""
    unit = ""
    for i in numU:
        if i in alphas:
            unit += i
        else:
            st += i
    return st, unit


def fraction_process(num):
    """
    分数带分数转浮点数，去除多余符号
    :param num:
    :return:
    """
    stamp = ""
    francium = ""
    count = 0
    for i in range(len(num)):
        if num[i] in '0123456789':
            stamp += num[i]
        if num[i] == "(":
            count = i
            break
    for i in range(count, len(num)):
        francium += num[i]
        if num[i] == ")":
            break
    if stamp == "":
        return eval(francium)
    else:
        return float(stamp) + eval(francium)


def texttoTloat(number):
    """
    将数字表里面的百分数、带分数、分数都转化成统一float
    :param number:
    :return:
    """
    num_list = []
    for num in number:
        if '%' in num:
            num_list.append(float(num[:-1])/100.0)
        elif '(' in num:
            enum, unit = split_num_unit(num)
            enum = fraction_process(enum)
            num_list.append(enum)
        else:
            enum, unit = split_num_unit(num)
            num_list.append(float(enum))
    return num_list


def suffix_equ(equation):
    """
    将公式转成后缀表达式
    :param equation:
    :return:
    """
    stack = []
    new_equ = []
    op_list = ['+', '-', '*', '/', '^']
    priority_op={'^':3, '/':2, '*':2, '+':1, '-':1}
    # 优先级低的先出栈
    for elem in equation:
        if elem == '(':
            stack.append(elem)
        elif elem == ')':
            while 1:
                op = stack.pop()
                if op == '(':
                    break
                else:
                    new_equ.append(op)
        elif elem in op_list:
            while 1:
                if stack == []:
                    break
                elif stack[-1] == '(':
                    break
                elif priority_op[elem] > priority_op[stack[-1]]:
                    break
                else:
                    op = stack.pop()
                    new_equ.append(op)
            stack.append(elem)
        else:
            new_equ.append(elem)
    while stack != []:
        new_equ.append(stack.pop())
    return new_equ


def change_pos(equation):
    """
    这是论文中提到的归一化的第二条规则，方程式模板中的数字标记的顺序应尽可能接近其在数字映射中的顺序
    对于第二条规则，事实上，数据集中并没有a+b+c+c-c的情况。
    :param equation:
    :return:
    """
    i = 0
    new_equ = []
    res = ['^']
    Len = len(equation)
    while i < Len:
        if i + 4 < Len and 'temp' in equation[i] and 'temp' in equation[i + 2] and '+' == equation[i + 1] and 'temp' in \
                equation[i + 4] and '+' == equation[i + 3]:
            if i - 1 > 0 and equation[i-1] in res+['-', '*', '/']:
                new_equ.append(equation[i])
                i += 1
                continue
            elif i+5<Len and equation[i+5] in res+['*', '/']:
                new_equ.append(equation[i])
                i += 1
                continue
            temp = [equation[i], equation[i+2], equation[i+4]]
            sort_temp = sorted(temp)
            new_equ += (sort_temp[0:1]+['+']+sort_temp[1:2]+['+']+sort_temp[2:3])
            i+=5
        elif i + 4 < Len and 'temp' in equation[i] and 'temp' in equation[i + 2] and '*' == equation[i + 1] and 'temp' in \
                equation[i + 4] and '*' == equation[i + 3]:
            if i - 1 > 0 and equation[i-1] in res:
                new_equ.append(equation[i])
                i += 1
                continue
            elif i+5<Len and equation[i+5] in res:
                new_equ.append(equation[i])
                i += 1
                continue
            temp = [equation[i], equation[i+2], equation[i+4]]
            sort_temp = sorted(temp)
            new_equ += (sort_temp[0:1]+['*']+sort_temp[1:2]+['*']+sort_temp[2:3])
            i+=5
        elif i+2 < Len and 'temp' in equation[i] and 'temp' in equation[i+2] and '+' == equation[i+1]:
            if i-1 > 0 and equation[i-1] in res+['-', '*', '/']:
                new_equ.append(equation[i])
                i+=1
                continue
            elif i+3 < Len and equation[i+3] in res+['*', '/']:
                new_equ.append(equation[i])
                i += 1
                continue
            temp = [equation[i], equation[i+2]]
            sort_temp = sorted(temp)
            new_equ += (sort_temp[0:1]+['+']+sort_temp[1:2])
            i+=3
        elif i+2 < Len and 'temp' in equation[i] and 'temp' in equation[i+2] and '*' == equation[i+1]:
            if i-1 > 0 and equation[i-1] in res:
                new_equ.append(equation[i])
                i+=1
                continue
            elif i+3 < Len and equation[i+3] in res:
                new_equ.append(equation[i])
                i += 1
                continue
            temp = [equation[i], equation[i+2]]
            sort_temp = sorted(temp)
            new_equ += (sort_temp[0:1]+['*']+sort_temp[1:2])
            i+=3
        else:
            new_equ.append(equation[i])
            i+=1
    return new_equ


def post_solver(post_equ):
    """
    计算后缀表达式的结果，通过栈来计算
    :param post_equ:
    :return:
    """
    stack = []
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        if elem not in op_list:
            op_v = elem
            #if '%' in op_v:
            #    op_v = float(op_v[:-1])/100.0
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


def solve_equation(equ_list):
    # if '=' in equ_list:
    #     equ_list = equ_list[2:]
    post_equ = suffix_equ(equ_list) # 通过后缀表达式把括号去掉
    ans = post_solver(post_equ)
    return ans


def ans_num_joint(ans):
    """
    将每个符号分开
    :param ans:
    :return:
    """
    i = 0
    new = []
    str_ = ''
    while i < len(ans):
        if ans[i].isdigit() or ans[i] in ['.','-']:
            str_ += ans[i]
        else:
            if str_ != '':
                new.append(str_)
                str_ = ''
            new.append(ans[i])
        i+=1
    return solve_equation(new)


def ans_fix(ans):
    """
    将ans转成float，处理百分数，分数
    :param ans:
    :return:
    """
    try:
        float(ans)
        return float(ans)
    except:
        if '%' in str(ans):
            return float(ans[:-1]) / 100
        if str(ans)[0] == '(' and str(ans)[-1] == ')':
            return ans_num_joint(ans)
    return -float('inf')


def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


train_data_list = get_math23k_data("./data/math23k_train.json")
test_data_list = get_math23k_data("./data/math23k_test.json")
sni_dict = get_json_data("./data/sni_DNS.json")

# ----------------------------Train-------------------------------------
for elem in train_data_list:
    #origin = elem['original_text']
    #print (sni_dict[elem['id']])
    elem['sni_text'] = sni_dict[elem['id']]['text']
    origin = elem['sni_text']
    origin_text = ' '.join(jieba.cut(origin, cut_all=False))
    elem['new_split'] = origin_text  # 这里不知道为什么要加一步这个？？？不是已经有segmented_text了吗

for elem in train_data_list:
    eid = elem['id']
    split_text = elem['segmented_text']
    text = split_text.split()
    equ = elem['equation']
    text_list, number_list, equ_list = mask_number(text, equ)
    num_list = texttoTloat(number_list)
    # print(num_list)
    # 情况5的处理
    if "千" in equ_list:
        equ_list = equ_list[: equ_list.index("千")]
    # print(equ_list)
    temp_equ = change_pos(equ_list)
    suffix_equ_list = suffix_equ(temp_equ)
    # print(temp_equ)
    if sni_dict[eid]['norm_template'] != '':
        suffix_equ_list = ['x','=']+sni_dict[eid]['norm_template']
    elem['target_template'] = temp_equ
    elem['target_norm_post_template'] = suffix_equ_list
    elem['text'] = ' '.join(text_list)
    elem['number_list'] = num_list
    elem['answer'] = float(ans_fix(elem['ans']))
    # print(ans_fix(elem['ans']))

train_shuffle = train_data_list[:]
random.shuffle(train_shuffle)
valid_set = train_shuffle[:1000]
train_set = train_shuffle[1000:]
write_data_json(train_set,"./data/new_train23k_processed.json")
write_data_json(valid_set,"./data/new_valid23k_processed.json")

# ----------------------------Test-------------------------------------
for elem in test_data_list:
    #origin = elem['original_text']
    #print (sni_dict[elem['id']])
    elem['sni_text'] = sni_dict[elem['id']]['text']
    origin = elem['sni_text']
    origin_text = ' '.join(jieba.cut(origin, cut_all=False))
    elem['new_split'] = origin_text  # 为什么要用sni里面的text

for elem in test_data_list:
    eid = elem['id']
    split_text = elem['segmented_text']
    text = split_text.split()
    equ = elem['equation']
    text_list, number_list, equ_list = mask_number(text, equ)
    num_list = texttoTloat(number_list)
    # print(num_list)
    # 情况5的处理
    if "千" in equ_list:
        equ_list = equ_list[: equ_list.index("千")]
    # print(equ_list)
    temp_equ = change_pos(equ_list)
    suffix_equ_list = suffix_equ(temp_equ)
    # print(temp_equ)
    if sni_dict[eid]['norm_template'] != '':
        suffix_equ_list = ['x','=']+sni_dict[eid]['norm_template']
    elem['target_template'] = temp_equ
    elem['target_norm_post_template'] = suffix_equ_list
    elem['text'] = ' '.join(text_list)
    elem['number_list'] = num_list
    elem['answer'] = float(ans_fix(elem['ans']))
    # print(ans_fix(elem['ans']))
test_set = test_data_list[:]
write_data_json(test_set,"./data/new_test23k_processed.json")
