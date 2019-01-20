import re
import pandas as pd

PRECEDING_DICT = ['number','numbers','digit','digits','integer','integers','consecutive']


def number_mapper(txt, eq_num_list):
    """
    Looks for numbers inside the text

    Args:
        txt: Original text
        eq_num_list: A list of numbers from the equations

    Returns:
        new_str: The original string with the numbers replaced with variables
        num_list: A list of all the numbers in the text
        var_list: A list of all the variables that were created

    """
    new_str = text2int(txt.lower())
    num_list = re.findall(r'[0-9/]+', new_str)
    var_list = []
    new_str = re.sub(r'[0-9/]+', '$N', new_str)
    original_eq_num_list = eq_num_list.copy()
    for possible_num in num_list:
        # real number
        if possible_num in eq_num_list:
            # Checks if the number is in the equation numbers
            i = original_eq_num_list.index(possible_num)
            new_str = re.sub(r'\$N', f'$n{i}', new_str, count=1)
            var_list.append(f'$n{i}')
            eq_num_list.remove(possible_num)
        else:
            # Else set it as variable
            new_str = re.sub(r'\$N', f'$M', new_str, count=1)
            var_list.append(f'$v')

    return new_str, num_list, var_list


def text2int(textnum):
    """
    Converts the textual number to numbers
    Args:
        textnum: Text with textual numbers

    Returns: Text with textual numbers (one,two,three) replaced with numeric numbers (1,2,3)

    """
    numwords = {}
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]

    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    scales = ["hundred", "thousand", "million", "billion", "trillion"]

    # numwords["and"] = (1, 0)
    for idx, word in enumerate(units):  numwords[word] = (1, idx)
    for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'twice': 2}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', '-')

    current = result = 0
    curstring = ""
    onnumber = False
    tokenized_text = textnum.split()
    for word,word_next in zip(tokenized_text,tokenized_text[1:]+[';']):
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            elif word_next in PRECEDING_DICT:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring


def list_number_mapper(eq_list):
    """

    Args:
        eq_list: List of equations for the textual problem

    Returns:
        new_eq_list: List of equations with variables instead of numbers
        numbers_list: All the numbers in the equations

    """
    new_eq_list = []
    numbers_list = []
    # Save the numbers that appear in the equations and replace them with the $N symbol
    for str in eq_list:
        if str[:4] == 'unkn':
            new_eq_list.append(str)
            continue
        new_str = text2int(str.lower())
        num_list = re.findall(r'[0-9/]+', new_str)
        new_str = re.sub(r'[0-9/]+', '$N', new_str)

        new_eq_list.append(new_str)
        numbers_list += num_list

    # Changes $N to $n0 $n1 etc
    for i in range(len(numbers_list)):
        for j, equation in enumerate(new_eq_list):
            new_eq_list[j], is_changed = re.subn(r'\$N', f'$n{i}', equation, count=1)
            if is_changed:
                break
    return new_eq_list, numbers_list


def number_parsing(equation_list, text):
    """

    Args:
        equation_list: List of the equations
        text: The text with the textual math equations

    Returns:
            equation list: a new list where numbers appear as $n0, $n1,...
            eq_num_list: values of the $n0, $n1,...
            text: the word math question with numbers as $n0, $n1,...
    """

    text = re.sub('[.,]', '', text).replace("-", " ")
    equation_list_template, eq_num_list = list_number_mapper(equation_list)
    tmp_eq_num_list = eq_num_list.copy()
    text, text_num_list, var_list = number_mapper(text, tmp_eq_num_list)
    new_equation_list = generate_new_equation(equation_list_template, eq_num_list, var_list)
    return new_equation_list, eq_num_list, text, var_list, text_num_list


def test_number_parsing(text):
    text = re.sub('[.,]', '', text).replace("-", " ")
    new_text_list, numbers_list = list_number_mapper([text])
    return new_text_list[0], numbers_list


def generate_new_equation(equation_list_template, eq_num_list, var_list):
    new_equations = [equation_list_template[0]]
    for equation in equation_list_template[1:]:
        # put back the old numbers from the equation
        for i in range(len(eq_num_list)):
            if f'$n{i}' not in var_list:
                equation = re.sub(f'\$n{i}', eq_num_list[i], equation)

        # finish
        new_equations.append(equation)

    return new_equations
