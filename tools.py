import re
from operator import add, sub, mul, truediv
from beartype import beartype


def Calendar(arg="", date=None):
    assert len(
        arg) == 0 or arg == " ", "Argument to the Calendar API should be empty."
    import datetime
    from calendar import day_name, month_name
    # datetime from date that is a string from a datetime object:
    if date is not None:
        now = datetime.datetime.strptime(date, '%Y-%m-%d')
    else:
        now = datetime.datetime.now()
    return f'Today is {day_name[now.weekday()]}, {month_name[now.month]} {now.day}, {now.year}'


def calend_parse(args: str):
    return args


def length_of_match(pattern, input_string):
    match = re.search(pattern, input_string)
    if match:
        return match.end() - match.start()
    else:
        return 0


def balance_parentheses(string):
    # While there exists "()", remove all occasions of "()" from the string
    while re.search(r'\(\)', string):
        string = re.sub(r'\(\)', '', string)

    # Find the starting substring that contains only "(" and ")":
    closing_at_start_count = length_of_match(r'^\)+', string)
    if closing_at_start_count > 0:
        string = string[closing_at_start_count:]

    opening_at_start_count = length_of_match(r'^\(+', string)

    opening_at_end_count = length_of_match(r'\(+$', string)
    if opening_at_end_count > 0:
        string = string[:-opening_at_end_count]
    closing_at_end_count = length_of_match(r'\)+$', string)

    remove = min(opening_at_start_count, closing_at_end_count)
    string = string[remove:-remove] if remove > 0 else string

    opening_count = string.count('(')
    closing_count = string.count(')')

    diff = opening_count - closing_count
    if diff > 0:
        string = string + ')' * diff
    else:
        string = '(' * (-diff) + string

    i = 0
    compensate = 0
    for char in string:
        if char == '(':
            i += 1
        elif char == ')':
            i -= 1
        if i < 0:
            compensate = min(i, compensate)

    string = '(' * abs(compensate) + string + ')' * abs(compensate)

    return string


@beartype
def calc_parse(args: str):
    multi_x_pattern = r'(?<=\d)x(?=\d)|(?<=\d)x(?!\d)|(?<!\d)x(?=\d)'
    args = re.sub(" ", "", args)
    args = re.sub(multi_x_pattern, "*", args)
    args = re.sub("÷", "/", args)
    args = re.sub("−", "-", args)
    args = re.sub("×", "*", args)
    args = re.sub(r'[^0-9+\-*/().]', '', args)

    args = balance_parentheses(args)

    return args


@beartype
def Calculator(input_query: str, extraargs=None, first=True, detail=False, inference=False):
    operators = {
        '+': add,
        '-': sub,
        '*': mul,
        '/': truediv
    }
    if first:
        # calc_preprocess_args(input_query) FOR INFERENCE
        # Strip whitespace
        input_query = input_query.replace(" ", "")
        input_query = balance_parentheses(input_query)

    # Handle expressions within parentheses
    while '(' in input_query:
        start = input_query.rindex('(')
        end = start + input_query[start:].index(')')
        sub_expr = input_query[start + 1:end]
        result, op_performed = Calculator(sub_expr, first=False, detail=True)
        first = not op_performed if first else first
        input_query = input_query[:start] + str(result) + input_query[end + 1:]

    try:
        number = float(input_query)
        if first and not inference:
            raise Exception("Useless API call. 1 digit is not a calculation.")
        else:
            return (number, False) if detail else number
    except ValueError as e:
        pass

    for c in operators.keys():
        left, operator, right = input_query.partition(c)
        if len(operator) > 0:
            if operator == '-' and len(left) == 0:
                answer = -Calculator(right, first=False)
            else:
                answer = round(operators[operator](Calculator(
                    left, first=False), Calculator(right, first=False)), 2)
            answer = str(answer) if first else answer
            return (answer, True) if detail else answer


def run_calculator_tests():
    test_cases = [
        # Basic arithmetic expressions
        ("2+3*4", 14.0),
        ("(2+3)*4", 20.0),
        ("(2-3)*4", 20.0),
        ("2+(3+4)*5", 37.0),
        ("1+2*3-4/2", 5.0),

        # Unbalanced parentheses at the start and end
        ("(2+3)*4)", 20.0),
        ("(1+2)*3)/2", 4.5),

        # Unbalanced parentheses in the middle
        ("2+((3+4)*5", 37.0),
        ("((1+2)*3)/2", 4.5),

        # Only opening parentheses at the start
        ("(((((2+3)*4", 20.0),

        # Only closing parentheses at the start
        ("))))2+3)*4", 20.0),

        # Mismatched parentheses
        (")2+3)*4(", 20.0),
        ("((2+3)*4))", 20.0),
        ("((1+2)*3)/2)))", 4.5),

        # No parentheses, basic arithmetic
        ("2+3", 5.0),
        ("3*4", 12.0),
        ("1-2", -1.0),
        ("4/2", 2.0),

        # Complex expression with unbalanced parentheses
        ("(2+(3*4", 14.0),
        ("(1+2)*(3/2", 4.5),
        ("(2+(3*4)-(5/2)+1", 12.5),

        # Tests that should fail
        ("(()))", None),
        ("()", None),
        ("(2))", None),
    ]

    for expression, expected_result in test_cases:
        try:
            result = Calculator(expression)
            result = float(result)
            assert result == expected_result, f"Expected {expected_result}, got {result}"
            print(f"Test passed: {expression} = {expected_result}")
        except Exception as e:
            if expected_result is None:
                print(
                    f"Test passed: {expression} raised exception as expected")
            else:
                print(f"Test failed: {expression}. Error: {e}")
                raise Exception(
                    f"CALCULATOR TESTS FAILED: {expression} with error {e}")


stopwords = None
searcher = None


def init_wikisearch():
    from pyserini.search.lucene import LuceneSearcher
    from nltk.corpus import stopwords as en_stopwords

    global searcher, stopwords
    searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
    stopwords = en_stopwords.words("english")

init_wikisearch()


@beartype
def wiki_parse(args: str):
    global stopwords
    # Only leave alphanumeric characters and spaces and hyphens
    args = re.sub(r'[^a-zA-Z0-9 \-]', ' ', args)
    # Remove double spaces:
    args = re.sub(r' +', ' ', args)
    # Remove stopwords:
    args = ' '.join([word for word in args.split() if word not in stopwords])
    # Lowercase:
    args = args.lower()
    # Remove leading and trailing spaces:
    args = args.strip()
    return args


def WikiSearch(term: str, args=None):
    # assert the term is not empty
    global searcher
    assert len(term) > 0, "Argument to the WikiSearch API should not be empty."
    hits = searcher.search(term)
    return hits[0].raw
