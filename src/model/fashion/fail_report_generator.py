import numpy  as np
import pandas as pd
import util  as ut


class FailReportGenerator:
    def __init__(self, tokenizer, all_set, test_set, test_dataset, targets, predictions, images_path):
        self.tokenizer    = tokenizer
        self.all_set      = all_set
        self.test_set     = test_set
        self.test_dataset = test_dataset
        self.targets     = np.concatenate(targets)
        self.predictions = np.concatenate(predictions)
        self.branch_mapping = {row['branch_seq']: (row['branch'], row['image_uri'], row['id']) for _, row in self.test_set[['branch', 'branch_seq', 'image_uri', 'id']].drop_duplicates().iterrows()}
        self.images_path = images_path

    def __call__(self):
        report = []
        for i, (t, p) in enumerate(zip(self.targets, self.predictions)):
            if t != p:
                text = self.tokenizer.to_text(self.test_dataset[i][0])
                text = text.split('[SEP]')[0].split('[CLS]')[1]

                true_row = self.branch_mapping[t]
                pred_row = self.branch_mapping[p]

                report.append({
                    'id': true_row[2],
                    'description': text,
                    'true_class':  true_row[0],
                    'true_image': true_row[1],
                    'pred_class': pred_row[0],
                    'pred_image': pred_row[1]
                })

        report = pd.DataFrame(report)

        print(f'Total Fails: {report.shape[0] / self.test_set.shape[0]:.2f}%')

        report['pred_image'] = report['pred_image'].apply(lambda x: ut.to_image_html(f'{self.images_path}/{x}'))
        report['true_image'] = report['true_image'].apply(lambda x: ut.to_image_html(f'{self.images_path}/{x}'))

        ut.display_html(report)

        return report