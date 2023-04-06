
# Standard.
import logging
import os
import threading

import torch

class PrefetchReader():
    def __init__(self, dataset_reader, num_refs = 1, prefetch_num = 4):
        self.dataset_reader = dataset_reader
        self.prefetch_num = prefetch_num
        self.num_refs = num_refs

        self.prefetched_data = {}
        self.threads = {}
        self.data_lock = threading.Lock()

    def __len__(self):
        return len(self.dataset_reader)

    def __getitem__(self, index):
        if index not in self.threads:
            prefetch_indexes = range(
                index,
                min(index + self.prefetch_num, len(self)))
            for index_prefetch in prefetch_indexes:
                self.threads[index_prefetch] = self.prefetch(index_prefetch)
        if index in self.threads:
            self.threads[index].join()
            self.data_lock.acquire()
            outs = self.prefetched_data[index]
            del self.prefetched_data[index]
            self.data_lock.release()
            del self.threads[index]
            prefetch_indexes = range(
                index + 1,
                min(index + 1 + self.prefetch_num, len(self)))
            for index_prefetch in prefetch_indexes:
                if index_prefetch not in self.threads:
                    self.threads[index_prefetch] = self.prefetch(
                        index_prefetch)
        else:
            logging.error('Index: {0} out of range {1}'.format(index, len(self)))
        return outs

    def prefetch(self, index):
        reader_thread = ReaderThread(
            self.prefetched_data, self.dataset_reader, index,
            self.data_lock, self.num_refs)
        reader_thread.start()
        return reader_thread

class ReaderThread(threading.Thread):
    def __init__(self, data, dataset_reader, index, lock, num_refs = 1):
        threading.Thread.__init__(self)
        self.data = data
        self.dataset_reader = dataset_reader
        self.index = index
        self.lock = lock
        self.num_refs = num_refs
    def run(self):
        data = {}
        data['image_path'] = self.dataset_reader.get_image_path(self.index)
        data['keypoints_path'] = self.dataset_reader.get_keypoints_path(
            self.index)
        outs = self.dataset_reader.get_image_pair(self.index, self.num_refs)
        data['img_target'] = outs[0]
        data['img_refs'] = outs[1]
        data['grid_target2refs'] = outs[2]
        self.lock.acquire()
        self.data[self.index] = data
        self.lock.release()

