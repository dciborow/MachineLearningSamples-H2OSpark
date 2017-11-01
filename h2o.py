import pip
pip.main(['install', 'colorama==0.3.8'])
pip.main(['install', 'h2o_pysparkling_2.2'])

import pyspark
import SparkFiles
import pysparkling, h2o
import os

from h2o.estimators.gbm import H2OGradientBoostingEstimator

os.environ["PYTHON_EGG_CACHE"] = "~/"
class timeit():
    from datetime import datetime
    def __enter__(self):
        self.tic = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(self.datetime.now() - self.tic))

sc = SparkContext.getOrCreate()

h2o_context = pysparkling.H2OContext.getOrCreate(spark)

import time

current_milli_time = lambda: int(round(time.time() * 1000)

with timeit():
    startLoad = current_milli_time()
    df = h2o.import_file(path="", pattern='part-00000')
    endload = current_milli_time()
    print(endload-startLoad))

    start = current_milli_time()
    y = 'C1'
    x = df.col_names
    x.remove(y)
    print("Response = " + y)
    print("Pridictors = " + str(x))

    train, valid, test = df.split_frame(ratios=[.8, .1])

    start = current_milli_time()
    gbm = H2OGradientBoostingEstimator(max_depth=2, ntrees=3)
    gbm.train(x=x, y =y, training_frame=train, validation_frame=valid)

    varImp = gbm.varimp()
    end = current_milli_time()
    print(end-start)
    varImp
    #logger.log('Best model accuracy',test_accuracy)
