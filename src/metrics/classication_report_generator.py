import pandas as pd
from sklearn.metrics import classification_report


class ClassicationReportGenerator:
    def classification_report_to_df(sef, report):
        rows = []
        for clazz, metrics in report.items():
            if type(metrics) != float and ' avg' not in clazz:
                metrics['class'] = clazz.lower()
                rows.append(metrics)

        df = pd.DataFrame(rows)

        selected = [df.columns[-1]]
        selected.extend(df.columns[:-1])
        return df[selected].reset_index(drop=True)

    def generate(self, target, prediction):
        report = classification_report(target, prediction, output_dict = True)
        return self.classification_report_to_df(report)