"""
   Copyright 2021 Dugal Harris - dugalh@gmail.com

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import itertools
from threading import Thread
import time
from tqdm import tqdm
import sys


class Spinner(Thread):
    """ Class to run a spinner in a separate thread. """
    def __init__(self, label='', interval=0.2, leave=False, **kwargs):
        Thread.__init__(self, **kwargs)
        self._label = label
        self._interval = interval
        self._run = True
        self._leave = leave

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()

    def run(self):
        self._run = True
        cursors_it = itertools.cycle('/-\|')
        while self._run:
            cursor = next(cursors_it)
            tqdm.write(self._label + cursor, end='\r')
            sys.stdout.flush()
            time.sleep(self._interval)
        if self._leave == False:
            tqdm.write('', end='\r')
        elif isinstance(self._leave, str):
            tqdm.write(self._label + self._leave, end='\n')
        sys.stdout.flush()

    def stop(self):
        self._run = False
