# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from components.utils import parse_args

# Edge Types
EDGE_TYPES = {
    'ASKS': ('user', 'asks', 'question'),
    'REV_ASKS': ('question', 'rev_asks', 'user'),
    'ANSWERS': ('user', 'answers', 'answer'),
    'REV_ANSWERS': ('answer', 'rev_answers', 'user'),
    'HAS': ('question', 'has', 'answer'),
    'REV_HAS': ('answer', 'rev_has', 'question'),
    'ACCEPTED_ANSWER': ('question', 'accepted_answer', 'answer'),
    'REV_ACCEPTED': ('answer', 'rev_accepted', 'question'),
    'SELF_LOOP': ('user', 'self_loop', 'user')
}

DATASET_TYPE_1 = 'STACK_OVERFLOW'
DATASET_TYPE_2 = 'ASK_REDDIT'

