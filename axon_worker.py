from axon_id.models import remove_axons_tq
from taskqueue import TaskQueue
import axon_id

tq = TaskQueue('sqs://EmilySkeletons', region_name="us-west-2", green=False)
tq.poll(
  lease_seconds = 1000,
  verbose = True, # print out a success message
)