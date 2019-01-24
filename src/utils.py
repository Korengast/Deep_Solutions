from typing import List

import wolframalpha
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, convert_xor, implicit_multiplication
from sympy import Symbol
from sympy.solvers import solve
from numpy import *
import numpy as np

IS_NUM_DICT = ['integer', 'consecutive']
TRANSFORMATION = standard_transformations + (implicit_multiplication, convert_xor,)
WOLF_CLIENT = wolframalpha.Client(app_id="23XUAT-H2875HHEEX")


# region Utilities

def solve_with_wolfram(input_str: str):
    sol_wolfram = WOLF_CLIENT.query(input_str.split('equ: ')[-1].replace(' ', ''))
    if 'Solutions' in sol_wolfram.details:
        sol_wolf = sol_wolfram.details['Solutions']
    elif 'Solution' in sol_wolfram.details:
        sol_wolf = sol_wolfram.details['Solution']
    elif 'Roots' in sol_wolfram.details:
        sol_wolf = sol_wolfram.details['Roots']
    elif 'Root' in sol_wolfram.details:
        sol_wolf = sol_wolfram.details['Root']
    else:
        return []

    results = [result.split('=') for result in sol_wolf.replace(' ', '').split(',')]
    return {k: float(eval(v)) for k, v in results}


def solve_eq_string(math_eq_format: List[str], integer_flag=False):
    """
    >>  ["unkn: x,y",
    >>  "equ: x + 3 = y",
    >>  "equ: 2*y + 12 = 5*x"]
    :param math_eq_format:
    :return:
    """
    kw_parser = dict(evaluate=True, transformations=TRANSFORMATION)
    do_wolfram = use_wolfram(math_eq_format[1:])
    var_str = math_eq_format[0].replace(' ', '').split('unkn:')[-1]
    sym_var = tuple()

    for v in var_str.split(','):
        var = Symbol(v)
        # var = Symbol(v, integer=integer_flag) if integer_flag else Symbol(v)
        sym_var += (var,)
    if not do_wolfram:
        parse_eq_list = []
        for eq in math_eq_format[1:]:
            # remove whitespaces and move to onesided equation
            rhs, lhs = eq.split('equ:')[-1].replace(' ', '').split('=')
            parse_eq_list += [parse_expr(lhs, **kw_parser) * -1 + parse_expr(rhs, **kw_parser)]

        if len(sym_var) <= 3 and '^' not in ';'.join(math_eq_format[1:]).replace('equ:', '').replace(' ', ''):
            sol = solve(parse_eq_list)
            if sol == []:
                do_wolfram = True
        else:
            do_wolfram = True
    if do_wolfram:
        eq_wolf_format = ';'.join(math_eq_format[1:]).replace('equ:', '').replace(' ', '')
        sol = solve_with_wolfram(eq_wolf_format)

    if isinstance(sol, dict):
        eval_sol = [parse_expr(v, **kw_parser).evalf(subs=dict(sol)) for v in var_str.split(',')]
        if sol == []:
            return []
        elif integer_flag and not all([y.is_integer for x, y in sol.items()]):
            return []
        else:
            return eval_sol
    elif isinstance(sol, list):
        eval_sol = []
        for cur_sol in sol:
            eval_itr = [parse_expr(v, **kw_parser).evalf(subs=cur_sol) for v in var_str.split(',')]
            if sol == []:
                continue
            elif integer_flag and not all([y.is_integer for x, y in cur_sol.items() if y.is_number]):
                continue
            else:
                eval_sol += [eval_itr]
        return eval_sol


def sparse_binary_jaccard(v1, v2):
    v1_nz = set(v1.nonzero()[1])
    v2_nz = set(v2.nonzero()[1])
    return len(v1_nz.intersection(v2_nz)) / len(v1_nz.union(v2_nz))


def is_number(query_text: str):
    query_text = query_text.lower()
    if any([t in query_text for t in IS_NUM_DICT]):
        return True
    else:
        return False


def parse_ans_col(ans: str):
    if ans == 'ans_no_result':
        return []
    else:
        ans.replace
        if '|' not in ans:
            pass


# endregion

def is_same_result(real_ans, pred_ans):
    if len(pred_ans) > 0 and isinstance(pred_ans, list) and isinstance(pred_ans[0], list):
        for cur_pred_ans in pred_ans:
            for cur_real_ans in real_ans:
                if are_close(cur_pred_ans, cur_real_ans):
                    return True
    else:
        for cur_real_ans in real_ans:
            if are_close(pred_ans, cur_real_ans):
                return True
    return False


def are_close(l1, l2):
    try:
        res = len(l1) == len(l2) and np.allclose(np.sort(l1).astype(float), np.sort(l2).astype(float), rtol=0.001)
        return res
    except Exception as e:
        # print(e)
        return False


def use_wolfram(equations):
    use_wolfram = False
    for equation in equations:
        if '<' in equation or '>' in equation:
            use_wolfram = True
        elif '=' not in equation:
            use_wolfram = True
    return use_wolfram


def get_real_answer(problem):
    if problem['ans'] == 'ans_no_result':
        solutions = []
    else:
        solutions = problem['ans'].replace(' ', '').replace(';', ',').replace('{', '(').replace('}', ')').replace('^',
                                                                                                                  '**').replace(
            '%', '').replace('or', '|').split('|')
        solutions = [eval(solution) for solution in solutions if
                     '=' not in solution and 'n' not in solution and 'x' not in solution]
    if len(solutions) == 0:
        solutions = [problem['ans_simple']]

    return solutions


def get_vocabulary(df, is_text=True):
    """
    :param df:
    :param is_text:
    :return: dict

    This function returns a dict as a vocabulary.
    For text, each word is a value in the vocabulary,
    for equations, each char is a value in the vocabulary
    """
    results = set()
    if is_text:
        df['text'].str.lower().str.split().apply(results.update)
    else:
        df['str_vars'].str.lower().apply(results.update)
        df['str_eqn'].str.lower().apply(results.update)
    return dict((w, i + 1) for (i, w) in enumerate(results))


def get_max(df, field_type):
    """

    :param df:
    :param field_type:
    :return: The maximal length of text or equation in the df
    """
    if field_type == 'text':
        df['length'] = df[field_type].str.split().apply(len)
    else:
        df['length'] = df[field_type].apply(len)
    return max(df['length'])


def get_varsAndEqn_str(df):
    """

    :param df:
    :return: Vars and equations as one string separated by commas
    """
    df['str_vars'] = df['equations'].apply(lambda l: l[0].replace('unkn: ', ''))

    def str_eqn(eqn_list):
        eqn_list = eqn_list[1:]
        eqn_str = ''
        for eqn in eqn_list:
            if len(eqn_str) != 0:
                eqn_str = eqn_str + ','
            eqn_str = eqn_str + eqn.replace('equ: ', '')
        return eqn_str

    df['str_eqn'] = df['equations'].apply(str_eqn)
    return df


def pad_and_vectorize(df, txt_length, var_length, eqn_length,
                      txt_vocab, eqn_vocab):
    """

    :param df:
    :param txt_length:
    :param var_length:
    :param eqn_length:
    :param txt_vocab:
    :param eqn_vocab:
    :return: X and y as vectorized and padded text (X) and equations (y)
    """
    def complete_txt(txt):
        vec_txt = np.zeros(txt_length)
        counter = 0
        if len(txt)>txt_length:
            txt = txt[-txt_length:]
        for word in txt.lower().split():
            if word in txt_vocab.keys():
                vec_txt[counter] = txt_vocab[word]
            else:
                vec_txt[counter] = 0
            counter += 1
        return vec_txt

    df['X'] = df['text'].apply(complete_txt)

    def create_y(row):
        vec_var = np.zeros(var_length)
        counter = 0
        for c in row['str_vars'].lower():
            if c in eqn_vocab.keys():
                vec_var[counter] = eqn_vocab[c]
            else:
                vec_var[counter] = 0
            counter += 1

        vec_eqn = np.zeros(eqn_length)
        counter = 0
        for c in row['str_eqn'].lower():
            if c in eqn_vocab.keys():
                vec_eqn[counter] = eqn_vocab[c]
            else:
                vec_eqn[counter] = 0
            counter += 1

        num_vars = len(row['ans_simple'])
        num_eqn = len(row['equations']) - 1

        return np.array([num_vars] + list(vec_var) + [num_eqn] + list(vec_eqn))

    df['y'] = df.apply(create_y, axis=1)

    return df

def to_wolfram_format(row):
    """

    :param row:
    :return: The predictions in a format could be solve using wolframalpha package
    """
    if len(row['var']) == 0:
        var_part = 'unkn: '
    else:
        var_part = 'unkn: '+row['var'][0]
    if len(row['eqn'])==0:
        eqn_part = ['equ: ']
    else:
        eqn_part = row['eqn'].split(",")
        eqn_part = ['eqn: '+ e for e in eqn_part]
    return [var_part] + eqn_part


