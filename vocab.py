#
# Copyright 2024 André Lucas Schlichting
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union

class Vocab:
    """Phone Vocabulary class"""

    def __init__(self, vocab_file) -> None:
        raise NotImplementedError

    def __len__(self):
        pass

    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Encode text to a list of integers

        Args:
            text (Union[str, List[str]]):

        Raises:
            TypeError: text must be str or list[str]

        Returns:
            Union[List[int], List[List[int]]]: list of integers of encoded text
        """
        pass
   
    def decode(
        self, indexes: Union[int, List[int]]
    ) -> Union[List[str], List[List[str]]]:
        """Decode list of integers to text

        Args:
            indexes (Union[int, List[int]]): list of integers of encoded text

        Raises:
            TypeError: indexes must be list[int] or int

        Returns:
            Union[List[str], List[List[str]]]: list of decoded text
        """
        pass
