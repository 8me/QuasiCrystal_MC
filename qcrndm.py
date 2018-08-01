#!/usr/bin/env python

import numpy as np
import quantumrandom as qtrndm
import time, threading
import json
import os.path

class Random(object):
    def __init__(self, p_bRecreateSeeds=False):
        #
        #np.random.seed(int(qtrndm.hex()[:8],16))
        np.random.seed(time.gmtime(0))
        #
        #self.xml_filepath = "./test.xml"
        #self.xml_document = None
        #self._init_xml_file()
        #self._safe_seed(100)

        #newScript = xmlFile.createElement("script")
        #print(self.xml_file.toprettyxml())

    def rand(self):
            return np.random.random()

    


    def _safe_seed(self, seed):
        pass
