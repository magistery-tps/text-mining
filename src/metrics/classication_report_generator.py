import pandas as pd
from sklearn.metrics import classification_report



def f_beta_score(precision, recall, beta):
    beta  = float(beta)
    return (1 + (beta**2)) * ( (precision*recall) / (((beta**2)*precision) + recall) )



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

    def generate(self, target, prediction, beta=1):
        report = classification_report(target, prediction, output_dict = True)

        report_df = self.classification_report_to_df(report)

        report_df[f'f{beta}-score'] = f_beta_score(report_df['precision'].astype(float), report_df['recall'].astype(float), beta)
        return report_df
