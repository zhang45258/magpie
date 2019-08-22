from magpie import Magpie


magpie = Magpie()
#train_dir = 'data/hep-categories'
train_dir = 'C:\\data\\Railway_Passenger_Transport'
#train_dir = 'C:\\data\\Railway_Passenger_Transport'
EMBEDDING_SIZE = 50
MIN_WORD_COUNT = 1
WORD2VEC_CONTEXT = 1
magpie.train_word2vec(train_dir, vec_dim=EMBEDDING_SIZE, MWC=MIN_WORD_COUNT, w2vc=WORD2VEC_CONTEXT)
magpie.save_word2vec_model('save/embeddings/'+train_dir[-3:]+'_'+str(EMBEDDING_SIZE)+'_'+str(MIN_WORD_COUNT)+'_'+str(WORD2VEC_CONTEXT), overwrite=True)
print(train_dir[-3:]+'_'+str(EMBEDDING_SIZE)+'_'+str(MIN_WORD_COUNT)+'_'+str(WORD2VEC_CONTEXT)+'   Success!!!')