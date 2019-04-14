from io import TextIOBase
from collections import Mapping
from tensorboardX import SummaryWriter

class StatisticsCollector(Mapping):
    """
    Specialized class used for the collection and export of statistics during training, validation and testing.

    Inherits :py:class:`collections.Mapping`, therefore it offers functionality close to a ``dict``.
    """

    def __init__(self):
        """
        Initialization - creates dictionaries for statistics and formatting.
        """
        super(StatisticsCollector, self).__init__()

        # Set default "output streams" to None.
        self.tb_writer = None
        self.csv_file = None

        # dict for the statistics to track and their formatting
        self.statistics = dict()
        self.formatting = dict()

    def add_statistic(self, key:str, formatting: str) -> None:
        """
        Add a statistic to collector.
        The value of associated to the key is of type ``list``.

        :param key: Key of the statistic.
        :type key: str

        :param formatting: Formatting that will be used when logging and exporting to CSV.
        :type formatting: str
        """
        self.formatting[key] = formatting

        # instantiate associated value as list.
        self.statistics[key] = list()

    def __getitem__(self, key: str):
        """
        Get statistics value for given key.

        :param key: Key to value in parameters.
        :type key: str

        :return: Statistics value list associated with given key.
        """
        return self.statistics[key]

    def __setitem__(self, key: str, value) -> None:
        """
        Adds value to the list of the statistic associated with a given key.

        :param key: Key to value in parameters.

        :param value: Value to append to the list associated with given key.
        """
        self.statistics[key].append(value)

    def __delitem__(self, key: str) -> None:
        """
        Delete the specified key.

        :param key: Key to be deleted.
        """
        del self.statistics[key]

    def __len__(self):
        """
        Returns the number of tracked statistics.
        """
        return self.statistics.__len__()

    def __iter__(self):
        """
        Iterator on the tracked statistics.
        """
        return self.statistics.__iter__()

    def empty(self):
        """
        Empty the list associated to the keys of the current statistics collector.
        """
        for key in self.statistics.keys():
            del self.statistics[key][:]

    def initialize_csv_file(self, log_dir: str, filename: str) -> TextIOBase:
        """
        Creates a new ``csv`` file and initializes it with a header produced on the base of statistics names.

        :param log_dir: Path to file.
        :type log_dir: str

        :param filename: Filename to be created.
        :type filename: str

        :return: File stream opened for writing.
        """
        header_str = ''

        # Iterate through keys and concatenate them.
        for key in self.statistics.keys():
            header_str += key + ","

        # Remove last coma and add \n.
        header_str = header_str[:-1] + '\n'

        # Open file for writing.
        self.csv_file = open(log_dir + filename, 'w', 1)
        self.csv_file.write(header_str)

        return self.csv_file

    def export_to_csv(self, csv_file=None):
        """
        Writes current statistics to ``csv_file`` using the indicated formatting for each key.

        :param csv_file: File stream opened for writing. Optional, defaults to ``self.csv_file``.
        """
        # use default if csv_file not indicated
        if csv_file is None:
            csv_file = self.csv_file

        # If it is still None - raise error
        if csv_file is None:
            raise FileNotFoundError('Please indicate a csv file with csv_file '
                                    'or instantiate one with initialize_csv_file(self, log_dir, filename).')

        # Iterate through values and concatenate them.
        values_str = ''
        for key, value in self.statistics.items():
            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')

            # Add last value to string using formatting.
            values_str += format_str.format(value[-1]) + ","

        # Remove last coma and add \n.
        values_str = values_str[:-1] + '\n'

        # write to csv file
        csv_file.write(values_str)

    def export_to_checkpoint(self) -> dict:
        """
        Exports the collected data into a dictionary using the associated formatting and returns it.
        """
        chkpt = {}

        # Iterate through key, values and format them.
        for key, value in self.statistics.items():

            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')

            # Collect last value and save it
            chkpt[key] = format_str.format(value[-1])

        return chkpt

    def export_to_string(self, additional_tag='') -> str:
        """
        Returns current statistics in the form of a string using the appropriate formatting.

        :param additional_tag: An additional tag to append at the end of the created string.
        :type additional_tag: str

        :return: String being the concatenation of the statistics names & values.
        """
        stat_str = ''

        # Iterate through keys and values and concatenate them.
        for key, value in self.statistics.items():
            stat_str += key + ' '

            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')

            # Add value to string using formatting.
            stat_str += format_str.format(value[-1]) + "; "

        # Remove last two element.
        stat_str = stat_str[:-2] + " " + additional_tag

        return stat_str

    def initialize_tensorboard(self, tb_writer: SummaryWriter) -> None:
        """
        Memorizes the writer which will be used with this collector.
        """
        self.tb_writer = tb_writer

    def export_to_tensorboard(self, tb_writer=None) -> None:
        """
        Exports the current statistics to tensorboard.

        :param tb_writer: TensorBoard writer, optional.
        :type tb_writer: :py:class:`tensorboardX.SummaryWriter`
        """
        # Get episode number.
        episode = self.statistics['episode'][-1]

        if tb_writer is None:
            tb_writer = self.tb_writer

        # If it is still None - raise error
        if tb_writer is None:
            raise NotImplementedError('Not Tensorboard writer found. Please pass one with tb_writer or '
                                      'instantiate one with initialize_tensorboard(self, tb_writer: SummaryWriter).')

        # Iterate through keys and values and concatenate them.
        for key, value in self.statistics.items():
            # Skip episode.
            if key == 'episode':
                continue
            tb_writer.add_scalar(key, value[-1], episode)
