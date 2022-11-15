import numpy  as np
import pandas as pd
import util  as ut
import random


class SampleReportGenerator:
    def __init__(self, test_set, targets, predictions, images_path):       
        self.test_set    = test_set
        self.targets     = np.concatenate(targets)
        self.predictions = np.concatenate(predictions)
        self.images_path = images_path

        data = self.test_set[['branch', 'branch_seq', 'image_uri', 'id']].drop_duplicates().iterrows()
        self.branch_mapping = {}
        for _, row in data:
            if row['branch_seq'] in self.branch_mapping:
                self.branch_mapping[row['branch_seq']].append((row['branch'], row['image_uri'], row['id']))
            else:
                self.branch_mapping[row['branch_seq']] = [(row['branch'], row['image_uri'], row['id'])]


    def __call__(self, pred_image_count=10):
        report = []
        for i, (target_class, pred_class) in enumerate(zip(self.targets, self.predictions)):
            pred_rows = self.branch_mapping[pred_class]
            selected_pred_rows = [pred_rows[random.choice(range(len(pred_rows)))] for _ in range(pred_image_count)]

            report_row = {
                'true.text'   : self.test_set._get_value(i, 'features'),
                'true.class'  : self.branch_mapping[target_class][0][0],
                'true.image'  : self.test_set._get_value(i, 'image_uri'),

                'pred.class'  : pred_rows[0][0]
            }

            for i, sr in enumerate(selected_pred_rows):
                report_row[f'pred.image.{i}'] = sr[1]

            report.append(report_row)

        report = pd.DataFrame(report)
        
        report = report.drop_duplicates().reset_index(drop=True)

        print(f'Total Fails: {report.shape[0] / self.test_set.shape[0]*100:.2f}%')


        for i in range(pred_image_count):
            report[f'pred.image.{i}'] = report[f'pred.image.{i}'].apply(lambda x: ut.to_image_html(f'{self.images_path}/{x}'))

        report['true.image'] = report['true.image'].apply(lambda x: ut.to_image_html(f'{self.images_path}/{x}'))
        
        return report