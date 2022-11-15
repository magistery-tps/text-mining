import pandas as pd
from .classication_report_generator import ClassicationReportGenerator


class MultiClassicationReportGenerator:
    def __init__(self):
        self.__generator = ClassicationReportGenerator()

    def generate(self, model_results, beta=1):
        reports = []
        for model_name, result in model_results.items():
            model_report = self.__generator.generate(
                target     = [v.lower() for v in result.target.values],
                prediction = [v.lower() for v in result.prediction.values],
                beta       = beta
            )
            model_report['model'] = model_name
            reports.append(model_report)
        return pd.concat(reports).reset_index(drop=True)