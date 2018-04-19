#!/usr/bin/env python
# coding=utf8

import logging

try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):

        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())

from sampling import (generate_random_design,  # noqa
                      generate_full_design,
                      generate_sample)
