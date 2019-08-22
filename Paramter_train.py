from magpie import Magpie
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


magpie = Magpie()
labels = ['1111', '1112', '1113', '1114', '1115', '1116', '1117', '1118', '1121', '1122', '1123', '1124', '1131', '1132', '1133', '1134', '1135', '1141', '1142', '1143', '1144', '1151', '1152', '1153', '1154', '1211', '1212', '1213', '1214', '1215', '1216', '1217', '1218', '1219', '1221', '1222', '1223', '1231', '1232', '1233', '1234', '1235', '1241', '1242', '1243', '1251', '1311', '1312', '1313', '1314', '1321', '1322', '1323', '1331', '1332', '1333', '1334', '1341', '1342', '1343', '1344', '1345', '1351', '1411', '1421', '1431', '1441', '15', '2111', '2112', '2113', '2114', '2115', '2116', '2117', '2121', '2122', '2123', '2124', '2131', '2132', '2133', '2134', '2141', '2142', '2143', '2144', '2145', '2146', '2147', '2148', '2149', '21410', '2151', '2152', '2153', '2154', '2155', '2156', '2161', '2162', '2163', '2164', '2165', '2166', '2167', '2168', '2171', '2172', '2173', '2174', '2175', '2176', '2177', '2178', '2179', '21710', '21711', '2181', '2182', '2183', '2184', '2185', '2186', '2187', '2188', '2191', '2192', '2193', '2194', '2195', '2196', '221', '222', '223', '224', '2311', '2312', '2313', '2314', '2315', '2316', '2321', '2322', '2323', '2324', '24', '31', '32', '33', '34', '41', '42', '43', '51', '52', '53', '54', '55', '56', '57', '58', '61', '7111', '7112', '7113', '7114', '7115', '7116', '7117', '7118', '7119', '71110', '71111', '7121', '7122', '7123', '7124', '7125', '7126', '7127', '7128', '7129', '7131', '7132', '7133', '7134', '7135', '7136', '7137', '7138', '7139', '71310', '71311', '71312', '7141', '7142', '7151', '721', '722', '723', '724', '7311', '7312', '7313', '7314', '7315', '7316', '7321', '7322', '7323', '7324', '7325', '7326', '7331', '7332', '7333', '7334', '7335', '7336', '734', '74']
magpie.train_word2vec('C:\\data\\Railway_Passenger_Transport', vec_dim=300, MWC=8, w2vc=6)
magpie.fit_scaler('data/hep-categories')
magpie.init_word_vectors('data/hep-categories')

'''
保存在验证集上最好的模型。
filename：字符串，保存模型的路径
monitor：需要监视的值
verbose：信息展示模式，0或1
save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
例如，当监测值为val_acc时，模式应为max，
当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
period：CheckPoint之间的间隔的epoch数
https://keras.io/zh/callbacks/#history
'''
checkpoint = ModelCheckpoint(filepath='save/model/best.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only='True',
                             mode='auto',
                             period=1)


'''
当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。
该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能提升，则减少学习率
monitor：被监测的量
factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
epsilon：阈值，用来确定是否进入检测值的“平原区”
cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
min_lr：学习率的下限

'''
reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.1,
                                      patience=5,
                                      verbose=1,
                                      mode='auto',
                                      epsilon=0.001,
                                      cooldown=5,
                                      min_lr=0)

'''
#调参
for optimizer in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
    for BATCH_SIZE in [16, 32, 64, 128, 256]:
        print(optimizer+str(BATCH_SIZE))
        magpie.train('data/hep-categories',
                     labels,
                     batch_size=BATCH_SIZE,
                     callbacks=[checkpoint, reduceLROnPlateau],
                     test_ratio=0.1,
                     epochs=60,
                     verbose=1,
                     optimizer=optimizer,
                     logdir='C:\\magpie-master\\trainlog\\' + optimizer + '_' + str(BATCH_SIZE) + '.txt'
                    )
'''
#形成最终模型
magpie.train('data/hep-categories',
                     labels,
                     batch_size=16,
                     callbacks=[checkpoint, reduceLROnPlateau],
                     test_ratio=0.0,
                     epochs=60,
                     verbose=1,
                     optimizer='Adam',
                  )
magpie.save_word2vec_model('save/embeddings/best', overwrite=True)
magpie.save_scaler('save/scaler/best', overwrite=True)
magpie.save_model('save/model/best.h5')