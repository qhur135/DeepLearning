# 라이브러리 불러오기
import tensorflow as tf 
from tensorflow import keras # tensorflow 쉽게 사용하도록 돕는 라이브러리
import numpy as np
import matplotlib.pyplot as plt # 숫자 이미지 보기위한 라이브러리 

# mnist 데이터셋 불러오기
mnist = keras.datasets.mnist

# 데이터셋 학습용, 테스트용으로 나누기
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# 학습용 데이터 형태 보기
print(x_train.shape)
print(x_train[0])
print(y_train[0])


# 데이터 전처리(0-1 사이의 숫자로 바꿈)
x_train = x_train/255
x_test = x_test/255

# 전처리 결과 확인
x_train[0]

# 입력층(784) - 은닉층(256) - 은닉층(128) - 은닉층(64) - 출력층(10)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), 
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
    ])

# 최적화 함수, 손실 함수 설정 + 평가 지표 설정 + 가중치 초기화(랜덤)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.summary()

# 모델 학습 - 5번 반복
model.fit(x_train,y_train,epochs=5) 

# 모델 평가 - 테스트 데이터 넣어보기
model.evaluate(x_test,y_test)

# 예측 - 0번째 숫자 이미지로 보기
plt.imshow(x_train[0],cmap='gray')
plt.show()

# 예측 - 0번째 숫자 예측하기1
print(model.predict(x_train[0].reshape(1,28,28)))

# 예측 - 0번째 숫자 예측하기1 (가장 높은 확률 알려줌)
print(np.argmax(model.predict(x_train[0].reshape(1,28,28))))

