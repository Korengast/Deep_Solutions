from utils import solve_eq_string, is_same_result, get_real_answer, is_number
import pandas as pd
from text_to_template import number_parsing, test_number_parsing

class ExampleModel():
    def __init__(self, **kwargs):
        pass

    def fit(self, df, y=None):

        # TODO
        raise NotImplementedError
        return self

    def predict(self, df):
        # TODO
        raise NotImplementedError
        return corpus_df

    def score(self, corpus_df, frac=1, verbose=False, use_ans=True, output_errors=False):

        def solve(problem):
            try:
                a = solve_eq_string(problem["predicted_equations"], integer_flag=is_number(problem["text"]))
                return a
            except Exception as e:
                return []

        corpus_df = self.predict(corpus_df)

        error_list = []
        correct, total = 0, 0
        for k,problem in corpus_df.sample(frac=frac).iterrows():
            pred_ans = solve(problem)

            if is_same_result([problem['ans_simple']], pred_ans):
                correct += 1
                error_list += [(k, ';'.join(problem['equations']).replace('equ:', ''),
                                ';'.join(problem['predicted_equations']).replace('equ:', ''), problem['text'],'1')]
            elif use_ans and is_same_result(get_real_answer(problem), pred_ans):
                correct += 1
                error_list += [(k, ';'.join(problem['equations']).replace('equ:', ''),
                                ';'.join(problem['predicted_equations']).replace('equ:', ''), problem['text'],'1')]
            else:
                error_list += [(k, ';'.join(problem['equations']).replace('equ:', ''),
                                ';'.join(problem['predicted_equations']).replace('equ:', ''), problem['text'],'0')]

            total += 1
            if verbose: print(correct,total,correct/total)

        if output_errors:
            return correct / total, pd.DataFrame(error_list, columns=['ind', 'equations', 'predicted_equations','text','correct'])
        else:
            return correct / total

