"""
G & T - embeddinG&Training data, a package for crowdsourcing image labels

Copyright (C) 2020 Popsa.
Author: Łukasz Kopeć

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import unittest

import pandas as pd

from gandt.data.analyse_labels_data import get_workers_reliability


class TestWorkersReliability(unittest.TestCase):

    def setUp(self) -> None:
        self.exp_a_unreliable = pd.DataFrame(
            {'worker_quality':
                 [{'a': 0, 'b': 1}] * 20 +
                 [{'a': 1, 'b': 1}] * 80
             })

        self.exp_a_reliable = pd.DataFrame(
            {'worker_quality':
                 [{'a': 1, 'b': 1}] * 100
             })

    def test_workers_reliability(self):
        workers_reliability = get_workers_reliability(self.exp_a_unreliable)
        self.assertEqual(
            'a' in list(workers_reliability.loc[
                workers_reliability['is_worker_reliable'], 'worker_id']),
            False
        )

    def test_previous_experiments(self):
        workers_reliability = get_workers_reliability(
            self.exp_a_unreliable, previous_experiments=[self.exp_a_reliable])
        self.assertEqual(
            'a' in list(workers_reliability.loc[
                workers_reliability['is_worker_reliable'], 'worker_id']),
            True
        )

    def test_number_of_experiments(self):
        workers_reliability = get_workers_reliability(self.exp_a_unreliable)
        self.assertEqual(list(workers_reliability['count']), [5, 5])

        workers_reliability = get_workers_reliability(
            self.exp_a_unreliable, no_labels_per_experiment=5)
        self.assertEqual(list(workers_reliability['count']), [20, 20])


if __name__ == '__main__':
    unittest.main()
