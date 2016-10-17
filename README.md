# Homework2 - Policy Gradient 
Please complete each homework for each team, and <br>
mention who contributed which parts in your report.


# Problem 1

>1. Use TensorFlow to construct a 2-layer neural network as stochastic policy.
>2. Assign the output of the softmax layer to the variable `probs`.

`h1 = tf.contrib.layers.fully_connected(self._observations, hidden_dim, activation_fn=tf.tanh)
probs = tf.contrib.layers.fully_connected(h1, out_dim, activation_fn=tf.nn.softmax)`

# Problem 2
>1. Trace the code above
>2. Currently, variable `self._advantages` represents accumulated discounted rewards
>from each timestep to the end of an episode
>3. Compute surrogate loss and assign it to variable `surr_loss`

`surr_loss = -tf.reduce_mean(log_prob * self._advantages, name="loss_op")`

# Problem 3
>1. Read the example provided in HW2_Policy_Graident.ipynb
>2. Uncomment below function and implement it.

`return lfilter([1], [1, -discount_rate], x[::-1], axis=0)[::-1]`

# Problem 4
>1. Variable `b` is the reward predicted by our baseline
>2. Use it to reduce variance and then assign the result to the variable `a`

`a = r - b`

# Problem 5
上圖中為Policy Gradient為使用Baseline(上圖)和沒使用Baseline(下圖)的 Average Return 值，觀察結果Standard Deviation 可以發現有使用Baseline的結果比較好，且收斂次數較少(約少10次)。並且我們可以利用線性函數預測狀態軌跡並擬合數據，來達到比較好的效果，因此在每次迭代運算，我們使用最新獲得的軌跡來作為預測訓練基準函數(LinearFeatureBaseline)，在這次的作業中的範例中，因只有正負一兩種情況，較好用線性預測，因此效果不錯。


# Problem 6
我們經過實驗發現，因為learning rate會受到獎勵值範圍影響，我們可以利用在計算梯度前進行加入Normalization來降低這個依賴使Gradient 趨於穩定。

## Participation
| Name | Do |
| :---: | :---: |
| 郭士鈞 | 撰修內容細節 |
| 黃冠諭 | 撰修內容細節 |
| 蘇翁台 | 撰寫主體綱要 |

