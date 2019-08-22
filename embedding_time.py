from magpie import Magpie
import time

count = 10
magpie = Magpie()
while (count <= 500):
    start = time.clock()
    magpie.train_word2vec('data/hep-categories', vec_dim=count)
    magpie.save_word2vec_model('save/embeddings/here'+str(count), overwrite=True)
    end = time.clock()
    runtime = end - start
    print(str(count)+','+str(runtime))
    file = open('save/embeddings/here.txt', 'a')
    file.write('\n'+str(count)+','+str(runtime))
    file.close()
    count = count+10