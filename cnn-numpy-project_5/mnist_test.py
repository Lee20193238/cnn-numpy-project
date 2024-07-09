import numpy as np
from nn.network import Network

from nn.flatten import Flatten

from nn.conv import Conv
from nn.fullyconnected import FullyConnected
from nn.activation import sigmoid, relu, mse, linear, cross_entropy
# from nn.optimizer import AdamOptimizer, SGDOptimizer
from nn.optimizer import AdamOptimizer, SGDOptimizer


import mnist_loader
from matplotlib import pyplot as plt
from nn.maxpooling import MaxPooling
import time

def accuracy(net, X, Y):
    a = (np.argmax(cross_entropy._softmax(net.forward(X)), axis=1) == np.argmax(Y, axis=1)) ## loss 수정
    return np.sum(a) / float(X.shape[0]) * 100.

def one_hot(x, size):
    a = np.zeros((x.shape[0], size))
    a[np.arange(x.shape[0]), x] = 1.
    return a


if __name__ == '__main__':

    # start_time_total = time.time()
    ###########################################################################
    # TODO: 네트워크 초기화  (필요에 따라 내용을 수정후 레포트 작성)
    ###########################################################################

    # 심플 MLP 예제

    lr = 0.01
    layers = [
        Flatten((28, 28, 1)), # fully connected 전에 입력을 1차원으로 만들어주는 flatten() 삽입
        FullyConnected((28*28, 100), activation=relu, optimizer = SGDOptimizer(),
                       weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(2.0 / (28*28))),
        FullyConnected((100, 50), activation=relu, optimizer = SGDOptimizer(),
                       weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(2.0 / (100.))),
        FullyConnected((50, 10), activation=linear, optimizer = SGDOptimizer(),
                       weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(2.0 / (50.)))
    ]

    # 심플 CNN 예제
    # lr = 0.01
    # layers = [
    #         Conv((5, 5, 1, 16), strides=1, activation=relu, optimizer=AdamOptimizer(),
    #              filter_init=lambda shp: np.random.normal(size=shp) * np.sqrt(2.0 / (28*28))),
    #         Conv((6, 6, 16, 32), strides=2, activation=relu, optimizer=AdamOptimizer(),
    #              filter_init=lambda shp:  np.random.normal(size=shp) * np.sqrt(2.0 / (16*24*24))),
    #         Conv((6, 6, 32, 64), strides=2, activation=relu, optimizer=AdamOptimizer(),
    #              filter_init=lambda shp: np.random.normal(size=shp) * np.sqrt(2.0 / (32*10*10))),
    #         Flatten((3, 3, 64)),
    #         FullyConnected((3*3*64, 256), activation=relu,
    #                    optimizer = AdamOptimizer(),
    #                    weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(2.0 / (3*3*64))),
    #         FullyConnected((256, 10), activation=linear,
    #                    optimizer = AdamOptimizer(),
    #                    weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(2.0 / (100.)))
    # ]


    # 네트워크 객체 생성
    net = Network(layers, lr=lr, loss= cross_entropy) ## 수정
    
    ###########################################################################
    # 데이터 가져오기 원본
    ###########################################################################
    (train_data_X, train_data_Y), v, (tx, ty) = mnist_loader.load_data('./data/mnist.pkl.gz')
    train_data_Y = one_hot(train_data_Y, size=10)
    ty = one_hot(ty, size=10)
    train_data_X = np.reshape(train_data_X, [-1, 28, 28, 1])
    tx = np.reshape(tx, [-1, 28, 28, 1])


    ###########################################################################
    # 데이터 가져오기
    ###########################################################################
    # (train_data_X, train_data_Y), v, (tx, ty) = mnist_loader.load_data('./data/mnist.pkl.gz')
    #
    # # 랜덤으로 10,000장 선택
    # random_indices = np.random.choice(train_data_X.shape[0], 10000, replace=False)
    # train_data_X = train_data_X[random_indices]
    # train_data_Y = train_data_Y[random_indices]
    #
    # train_data_Y = one_hot(train_data_Y, size=10)
    # ty = one_hot(ty, size=10)
    # train_data_X = np.reshape(train_data_X, [-1, 28, 28, 1])
    # tx = np.reshape(tx, [-1, 28, 28, 1])
    ###########################################################################
    # TODO: 네트워크 학습  (필요에 따라 내용을 수정후 레포트 작성)
    ###########################################################################
    loss = []
    total_iter = 1000
    batch_size = 100

    for iter in range(total_iter):
        # start_time_iter = time.time() # 추가
        shuffled_index = np.random.permutation(train_data_X.shape[0])

        batch_train_X = train_data_X[shuffled_index[:batch_size]]
        batch_train_Y = train_data_Y[shuffled_index[:batch_size]]
        net.train_step((batch_train_X, batch_train_Y))
        loss.append(np.sum(cross_entropy.compute(net.forward(batch_train_X), batch_train_Y))) ## 수정

        if iter % 10 == 0 and iter > 1:
            print('Calculate accuracy over all test set (시간 소요):' + time.ctime())
            print('Accuracy over all test set %f' % accuracy(net, tx, ty))
            # end_time_iter = time.time()  # 각 iteration 끝난 시간 기록 추가!
            # print('Time for iteration {}: {:.2f} seconds'.format(iter, end_time_iter - start_time_iter)) # 추가!
        elif iter % 10 == 0:
            print('Iteration: %d, loss : %f' % (iter, loss[-1]))


    ###########################################################################
    # 마지막 결과 출력
    ###########################################################################
    # end_time_total = time.time()  # 학습 끝난 시간 기록 추가!
    print('#### 학습 종료 #####')
    print('Calculate accuracy over all test set (시간 소요)' + time.ctime())
    test_acc = accuracy(net, tx, ty)
    print('Accuracy over all test set %.2f' % test_acc)
    # print('Total training time: {:.2f} seconds'.format(end_time_total - start_time_total)) ## 추가!


    plt.plot(range(total_iter), loss)
    plt.title('Test accuracy: %.2f' % test_acc)
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['training loss'], loc='upper left')
    plt.show()

