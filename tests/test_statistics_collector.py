import os
from unittest import TestCase
from training.statistics_collector import StatisticsCollector


class TestDecoderLayer(TestCase):
    def test(self):

        stat_col = StatisticsCollector()

        # add some statistics and formatting
        stat_col.add_statistic('loss', '{:12.10f}')
        stat_col.add_statistic('episode', '{:06d}')
        stat_col.add_statistic('acc', '{:2.3f}')

        # add some dummy values
        stat_col['episode'] = 0
        stat_col['loss'] = 0.7
        stat_col['acc'] = 100

        # initialize a csv file
        csv_file = stat_col.initialize_csv_file('./', 'collector_test.csv')

        # write to file
        stat_col.export_to_csv(csv_file)

        # read csv file
        with open('collector_test.csv') as f:
            csv_content = f.read()

        # assert csv content matches the expected one
        self.assertEqual(str(csv_content), "loss,episode,acc\n0.7000000000,000000,100.000\n")

        # close file
        csv_file.close()

        # delete csv file
        os.remove('collector_test.csv')

        # add some values
        stat_col['episode'] = 1
        stat_col['loss'] = 0.7
        stat_col['acc'] = 99.3

        stat_col.add_statistic('seq_length', '{:2.0f}')
        stat_col['seq_length'] = 5

        export_str = stat_col.export_to_string('[Training]')

        self.assertEqual(export_str, "loss 0.7000000000; episode 000001; acc 99.300; seq_length  5 [Training]")

        # finally empty stat col
        stat_col.empty()

        for k in stat_col:
            self.assertEqual(stat_col[k], [])
