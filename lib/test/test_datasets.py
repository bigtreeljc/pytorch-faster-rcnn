import unittest
from datasets import pascal_voc
from torchloop.util import tl_logging
logger = tl_logging.tl_logger(tl_logging.DEBUG, True)

class test(unittest.TestCase):
    def test1(self):
        d = pascal_voc.pascal_voc("trainval", "2007")
        roidb = d.roidb
        logger.debug("len roidb {}".format(len(roidb)))
        logger.debug("name {} n_images {}".format(
            d.name, d.num_images))
        logger.debug("classes are\n{}".format(
            d.classes))

if __name__ == "__main__":
    unittest.main()
