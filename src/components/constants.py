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