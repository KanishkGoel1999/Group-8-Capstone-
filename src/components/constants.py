DATASET_TYPE_1 = 'STACK_OVERFLOW'
DATASET_TYPE_2 = 'ASK_REDDIT'

NODE_TYPES = {
    'STACK_OVERFLOW': ['user', 'question', 'answer'],
    'ASK_REDDIT': ['author', 'post', 'comment']
}

EDGE_TYPES = {
    'STACK_OVERFLOW': [
        ('user', 'asks', 'question'),
        ('question', 'rev_asks', 'user'),
        ('user', 'answers', 'answer'),
        ('answer', 'rev_answers', 'user'),
        ('question', 'has', 'answer'),
        ('answer', 'rev_has', 'question'),
        ('question', 'accepted_answer', 'answer'),
        ('answer', 'rev_accepted', 'question'),
        ('user', 'self_loop', 'user')
    ],
    
    'ASK_REDDIT': [
        ('author', 'wrote_post', 'post'),
        ('post', 'wrote_post__reverse', 'author'),
        ('author', 'wrote_comment', 'comment'),
        ('comment', 'wrote_comment__reverse', 'author'),
        ('post', 'has_comment', 'comment'),
        ('comment', 'has_comment__reverse', 'post'),
        ('author', 'self_loop', 'author')
    ]
}

