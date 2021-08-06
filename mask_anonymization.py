# -*- coding: utf-8 -*-
"""
Anonymize tweets by replacing words to maximize the anonymization 

@author: elizabeth
"""

from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("Hello I'm a [MASK] model.")