#!/usr/bin/env python
# coding: utf-8

# ## 1-1. 머신러닝 훑어보기

# ### 1. 머신러닝 모델의 평가
# - 훈련 데이터: 머신러닝 모델 학습 용도
# - 검증 데이터: 모델이 훈련데이터에 과적합 되었는지 판단하거나 하이퍼파라미터의 조정을 위한 용도
#   - 하이퍼파라미터: 사람이 값 지정, ex) 학습률, 뉴런의 수, 층의 수
#   - 매개변수: 학습과정에서 얻어지는 값, ex) 가중치, 편향
# - 테스트 데이터: 머신러닝 모델의 성능 평가 용도
# 
# 
# ### 2. Classification & Regression
# - 종류: 이진 분류, 다중 클래스 분류, 회귀 문제
# 
# 
# ### 3. Confusion matrix
# 1. 정밀도(Precision): 모델이 True로 분류한 것 중 실제 True인 비율 $(= \frac{TP}{TP+FP})$
# 
# 2. 재현율(Recall): 실제 True인 것 중 모델이 True로 예측한 비율 $(= \frac{TP}{TP+FN})$
# 
# 3. 정확도(Accuracy): 전체 데이터 중 정답 비율 $(= \frac{TP + TN}{TP+FN+FP+TN})$
# 
# 
# ### 4. Optimizer: Gradient Descent
# $$cost(w,b) = \frac{1}{n} \sum_{i=1}^n [ y^{(i)} - H(x^{(i)}) ]^2$$
# 
# $$w, b \longrightarrow minimize cost(w,b)$$
# 
# $$w := w - \alpha \frac{d}{dw}cost(w)$$

# ## 1-2. 자동 미분과 선형 회귀 실습

# ### 1. 자동 미분
# - 임의로 $2w^2 + 5$라는 식을 세운 후 $w$에 대해 미분

# In[ ]:


import tensorflow as tf

# 변수 생성 (object 형태)
w = tf.Variable(2.)
print(w)

# 실제 사용
tf.print(w)


# In[ ]:


def f(w):
  return 2*(w**2) + 5

# 연관된 연산을 "테이프"에 기록 
with tf.GradientTape() as tape:
  z = f(w)

# 오차 역전파 (후진 모드 자동 미분)
gradients = tape.gradient(z, [w])
print(gradients)


# ### 2. 자동 미분을 이용한 선형 회귀 구현

# In[ ]:


# 학습될 가중치 변수 선언
w = tf.Variable(4.0)
b = tf.Variable(1.0)

# 가설을 함수로서 정의
def hypothesis(x):
  return w*x + b

x_test = [3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())


# In[ ]:


# 평균 제곱 오차를 손실 함수로서 정의 (두 개의 차이값을 제곱 후 평균)
def mse_loss(y_pred, y):
  
  # reduce_mean(): 특정 차원을 제거하고 평균을 구함 
  return tf.reduce_mean(tf.square(y_pred - y))

# 데이터
x = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적

# 옵티마이저: 경사 하강법(Stochastic Gradient Descent, SGD), 학습률 = 0.01
optimizer = tf.optimizers.SGD(0.01)


# In[ ]:


for i in range(301):

  with tf.GradientTape() as tape:
    # 현재 파라미터에 기반한 입력 x에 대한 예측값을 y_pred
    y_pred = hypothesis(x)

    # 평균 제곱 오차를 계산
    cost = mse_loss(y_pred, y)

  # 손실 함수에 대한 파라미터의 미분값 계산
  gradients = tape.gradient(cost, [w, b])

  # 파라미터 업데이트 (loss 작아지는 방향으로)
  optimizer.apply_gradients(zip(gradients, [w, b]))

  if i % 10 == 0:
    print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))


# In[ ]:


# 학습된 w와 b에 대해 임의 입력을 넣었을 경우의 예측값
x_test = [3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())


# ### 3. 케라스로 구현하는 선형 회귀

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적

# Sequential로 model이라는 이름의 모델 생성
model = Sequential()

# 단순 선형 회귀이므로 출력 y의 차원은 1. 입력 x의 차원(input_dim)은 1 / 선형 회귀이므로 activation은 'linear'
model.add(Dense(1, input_dim=1, activation='linear'))

# sgd는 경사 하강법을 의미. 학습률(learning rate, lr)은 0.01.
sgd = optimizers.SGD(learning_rate = 0.01)

# 손실 함수(Loss function)은 평균제곱오차 mse를 사용합니다.
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

# 주어진 x와 y데이터에 대해서 오차를 최소화하는 작업을 300번 시도합니다.
model.fit(x, y, epochs=300)


# In[ ]:


# 최종적으로 선택된 오차를 최소화하는 직선 (각 점은 실제값, 직선은 오차를 최소화하는 w와 b의 값을 가지는 직선)
plt.plot(x, model.predict(x), 'b', x, y, 'k.')


# In[ ]:


# 9시간 30분을 공부하였을 때의 시험 성적 예측
print(model.predict([9.5]))


# ## 1-3. 로지스틱 회귀

# ### 1. 시그모이드 함수
# 
# $$H(x) = \frac{1}{1+e^{-(wx+b)}} = sigmoid(wx+b) = \sigma(wx+b)$$
# 
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# w = 1, b = 0 가정 
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[0.0,1.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


# In[ ]:


def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # w의 값이 0.5일때
plt.plot(x, y2, 'g') # w의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--') # w의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


# In[ ]:


def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
plt.plot(x, y2, 'g') # x + 1
plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


#       w값이 커지면 경사가 커지고, b값이 커지면 1에 수렴

# ### 2. 비용 함수
# - 로지스틱 회귀의 경우, 평균 제곱 오차를 비용 함수로 사용하면 Local Minimum에 빠질 확률이 높으므로 사용하지 않음
# 
# - 목적 함수 (objective function)
# 
# $$J(w) = \frac{1}{n} \sum_{i=1}^n cost(H(x^{(i)}), y^{(i)})$$
# 
# - 로지스틱 회귀의 비용 함수: 크로스 엔트로피 (Cross Entropy)
# 
# $$cost(H(x), y) -[ylog H(x) + (1 - y)log (1 - H(x))]$$
# 
# $$J(w) = - \frac{1}{n} \sum_{i=1}^n [y^{(i)}log H(x^{(i)}) + (1 - y^{(i)})log (1 - H(x^{(i)}))]$$
# 
#       실제값이 1일 때, 예측값인 H(x)의 값이 1이면 cost는 0으로, 예측값인 H(x)의 값이 0이면 cost는 무한대로 발산

# ### 3. 로지스틱 회귀 실습

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 숫자 10부터 1

# Sequential로 model이라는 이름의 모델 생성
model = Sequential()

# 1개의 실수 x로부터 1개의 실수인 y를 예측하는 맵핑 관계 / activation은 'sigmoid'
model.add(Dense(1, input_dim=1, activation='sigmoid'))

# 옵티마이저 : 경사 하강법
sgd = optimizers.SGD(learning_rate = 0.01)

# 손실 함수(Loss function)은 크로스 엔트로피 함수(binary_crossentropy)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

# 에포크: 200
model.fit(x, y, epochs=200)


# In[ ]:


plt.plot(x, model.predict(x), 'b', x,y, 'k.')


# In[ ]:


print(model.predict([-3, -2, -1, -0.5, 0.5, 1, 2, 3]))


# ## 1-4. 다중 입력에 대한 실습

# ### 1. 다중 선형 회귀
# 
# $$H(X) = w_1x_1 + w_2x_2 + w_3x_3 + b, \quad X = [x_1, x_2, x_3]$$

# In[ ]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 중간 고사, 기말 고사, 가산점 점수
X = np.array([[70,85,11], [71,89,18], [50,80,20], [99,20,10], [50,10,10]]) 
y = np.array([73, 82 ,72, 57, 34]) # 최종 성적

model = Sequential()

# 입력의 차원 = 3
model.add(Dense(1, input_dim=3, activation='linear'))

sgd = optimizers.SGD(learning_rate = 0.0001)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, y, epochs=2000)


# In[ ]:


# 학습된 모델에 입력 X에 대한 예측
print(model.predict(X))


# ### 2. 다중 로지스틱 회귀
# 
# $$H(X) = sigmoid(w_1x_1 + w_2x_2 + b), \quad X = [x_1, x_2]$$

# In[ ]:


X = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])
y = np.array([0, 0, 0, 1, 1, 1])

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(X, y, epochs=2000)


# In[ ]:


# 입력의 합이 2 이상인 경우에는 전부 0.5 넘음 
print(model.predict(X))


# ### 3. 인공 신경망 다이어그램 
# 
# $$y = sigmoid(w_1x_1 + w_2x_2 + \ldots + w_nx_n + b) = \sigma(w_1x_1 + w_2x_2 + \ldots + w_nx_n + b)$$

# ## 1-5. 벡터와 행렬 연산

# ### 1. 텐서(Tensor)

# In[ ]:


import numpy as np

# 0차원 텐서 (스칼라)
d = np.array(5)
print('텐서의 차원 또는 축(axis)의 개수 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)


# In[ ]:


# 1차원 텐서 (벡터)
d = np.array([1, 2, 3, 4])
print('텐서의 차원 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)


# In[ ]:


# 2차원 텐서 (3행 4열의 행렬)
d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('텐서의 차원 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)


# In[ ]:


# 3차원 텐서 (다차원 배열)
d = np.array([
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [19, 20, 21, 22, 23], [23, 24, 25, 26, 27]]
            ])
print('텐서의 차원 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)


# ### 2. 벡터와 행렬의 연산

# In[ ]:


# 벡터의 연산 
A = np.array([8, 4, 5])
B = np.array([1, 2, 3])
print('두 벡터의 합 :',A + B)
print('두 벡터의 차 :',A - B)
print('두 벡터의 내적 :',np.dot(A, B))


# In[ ]:


# 행렬의 연산
A = np.array([[1, 3],[2, 4]])
B = np.array([[5, 7],[6, 8]])
print('두 행렬의 합 :')
print(A + B)
print('두 행렬의 차 :')
print(A - B)
print('두 행렬의 행렬곱 :')
print(np.matmul(A, B))


# ## 1-6. 소프트맥스 회귀

# ### 1. 다중 클래스 분류
# 
# - 3개 이상의 선택지 중 하나를 고르는 문제 (확률의 총 합이 1인 예측값을 얻어 이 중 확률값이 가장 높은 것으로 예측)

# ### 2. 소프트맥스 함수
# 
#   - $z_i$: k차원 벡터에서 i번째 원소 \\
#   - $p_i$: i번째 클래스가 정답일 확률
# 
# $$p_i = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}} \quad for \quad i = 1, 2, \ldots, k$$
# 
# 
#   - z = [$z_1, z_2, z_3$] 이면, $softmax(z) = [p_1, p_2, p_3]$

# ### 3. 원-핫 벡터의 무작위성
# 
# 정수 인코딩과 달리 원-핫 인코딩은 분류 문제 모든 클래스 간의 관계를 균등하게 분배하므로 다중 클래스 분류 문제에 적절함

# ### 4. 비용 함수
# 
# - 크로스 엔트로피 함수 \\
#   $y_j$: 실제값 원-핫 벡터의 j번째 인덱스 \\
#   $p_j$: 샘플 데이터가 j번째 클래스일 확률 $(= \hat{y_j})$
# 
# - 설명: $c$가 실제값 원-핫 벡터에서 1을 가진 원소의 인덱스라고 한다면, $p_c = 1$은 $\hat{y}$가 $y$를 정확하게 예측한 경우가 된다. 이를 식에 대입해보면 $-1 log(1) = 0$이 되기 때문에, 결과적으로 $\hat{y}$가 $y$를 정확하게 예측한 경우의 크로스 엔트로피 함수의 값은 0이 된다.

# ## 1-7. 소프트맥스 회귀 실습

# ### 1. 아이리스 품종 데이터에 대한 이해

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/06.%20Machine%20Learning/dataset/Iris.csv", filename="Iris.csv")

data = pd.read_csv('Iris.csv', encoding='latin1')

print('샘플의 개수 :', len(data))
data


# In[2]:


# 중복을 허용하지 않고, 있는 데이터의 모든 종류를 출력
print("품종 종류:", data["Species"].unique(), sep="\n")


# In[3]:


# 축에 '|' 표시하고 색 조정
sns.set(style="ticks", color_codes=True)

# pairplot : 데이터프레임을 입력으로 받아 각 열의 조합에 따라서 산점도 모두 표시 
# hue: Species 기준으로 나눠 그리기 / palette: 테마 색상 바꾸기 
g = sns.pairplot(data, hue="Species", palette="husl")


# In[7]:


# 각 종과 특성에 대한 연관 관계
sns.barplot(data = data, x = "Species", y = "SepalWidthCm", errorbar = None)


# In[10]:


# 150개의 샘플 데이터 중에서 Species열에서 각 품종이 몇 개있는지 확인
data['Species'].value_counts().plot(kind = 'bar')


# In[11]:


# 정수 인코딩: Iris-virginica는 0, Iris-setosa는 1, Iris-versicolor는 2가 됨.
data['Species'] = data['Species'].replace(['Iris-virginica','Iris-setosa','Iris-versicolor'],[0,1,2])
data['Species'].value_counts().plot(kind='bar')


# In[14]:


# X 데이터. 특성은 총 4개.
data_X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

# Y 데이터. 예측 대상.
data_y = data['Species'].values

print(data_X[:5])
print(data_y[:5])


# In[15]:


# 훈련 데이터와 테스트 데이터를 8:2로 나눈다.
(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)

# 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train[:5])
print(y_test[:5])


# ### 2. 소프트맥스 회귀

# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))

# 이진 분류 문제에서는 binary_crossentropy / 다중 클래스 분류 문제에서는 categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 전체 데이터에 대한 훈련 횟수는 200
# validation_data=(X_test, y_test): 실제 훈련에는 사용하지 않으면서 (가중치 업데이트 안함) 매 훈련 마다 테스트 데이터에 대한 정확도를 출력
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test))


#       - accuracy: 훈련 데이터에 대한 정확도
#       - val_accuracy: 테스트 데이터에 대한 정확도

# In[22]:


# 한 번 에포크에 따른 정확도를 그래프로 출력
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()


# In[25]:


# 테스트 데이터의 정확도를 측정 ([0]: loss, [1]: accuracy)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

