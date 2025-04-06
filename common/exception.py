# for reporting syntax errors when running rule on dummy data
class SyntaxException(Exception):
    def __init__(self, message, rule_path):
        super().__init__(message)
        self.rule_path = rule_path


# for reporting runtime errors when running rule on real data
class RuntimeException(Exception):
    def __init__(self, message, df, rule_path):
        super().__init__(message)
        self.df = df
        self.rule_path = rule_path


# for timeout
class TimeoutException(Exception):
    pass
